import os
import pickle
from datetime import datetime

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import CBERS4MUXDataset

LEARNING_RATE = 1e-4
TRIPLET_WEIGHT = 0.1
TRIPLET_MARGIN = 0.2
TRIPLET_SAMPLES = 2048
EMBEDDING_DIM = 32


class CBERS4MUXSiamese2BranchDataset(Dataset):
    """Retorna RGB e NIR+indices separados, dividindo a imagem em grade de crop_factor x crop_factor."""

    def __init__(self, red_image_paths, green_image_paths, blue_image_paths,
                 nir_image_paths, mask_paths, crop_factor=1, indices_to_add=None,
                 transform=None, mask_invert='auto'):
        self.crop_factor = crop_factor
        self.base = CBERS4MUXDataset(
            red_image_paths=red_image_paths,
            green_image_paths=green_image_paths,
            blue_image_paths=blue_image_paths,
            nir_image_paths=nir_image_paths,
            mask_paths=mask_paths,
            indices_to_add=indices_to_add,
            transform=transform,
            mask_invert=mask_invert,
        )

    def __len__(self):
        return len(self.base) * (self.crop_factor * self.crop_factor)

    def __getitem__(self, idx):
        croppings = self.crop_factor * self.crop_factor
        base_idx = idx // croppings
        crop_idx = idx % croppings

        image, mask = self.base[base_idx]

        # image: [B, G, R, NIR, NDVI, NDWI, GNDVI]
        rgb = image[:3, :, :]
        spectral = image[3:, :, :]

        if self.crop_factor > 1:
            r = crop_idx // self.crop_factor
            c = crop_idx % self.crop_factor

            h_size = image.shape[1] // self.crop_factor
            w_size = image.shape[2] // self.crop_factor

            h_slice = slice(r * h_size, (r + 1) * h_size if r < self.crop_factor - 1 else None)
            w_slice = slice(c * w_size, (c + 1) * w_size if c < self.crop_factor - 1 else None)

            rgb = rgb[:, h_slice, w_slice]
            spectral = spectral[:, h_slice, w_slice]
            mask = mask[:, h_slice, w_slice]

        return rgb, spectral, mask


class Siamese2BranchNet(nn.Module):
    """Rede Siamesa 2 branches: RGB + NIR/Indices, sem hidrografia."""

    def __init__(self, encoder_name: str = "efficientnet-b7"):
        super().__init__()

        self.branch_rgb = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=256,
            activation=None,
        )

        self.branch_spectral = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=4,
            classes=256,
            activation=None,
        )
        self._init_branch_spectral_from_rgb()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.embedding_head = nn.Conv2d(128, EMBEDDING_DIM, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _init_branch_spectral_from_rgb(self):
        state_dict_rgb = self.branch_rgb.state_dict()
        state_dict_spec = self.branch_spectral.state_dict()

        for key in state_dict_spec.keys():
            if key in state_dict_rgb:
                rgb_param = state_dict_rgb[key]
                spec_param = state_dict_spec[key]

                if rgb_param.shape == spec_param.shape:
                    state_dict_spec[key] = rgb_param
                    continue

                if (len(rgb_param.shape) == 4 and len(spec_param.shape) == 4
                        and rgb_param.shape[0] == spec_param.shape[0]
                        and rgb_param.shape[2:] == spec_param.shape[2:]
                        and rgb_param.shape[1] == 3
                        and spec_param.shape[1] == 4):
                    adapted = spec_param.clone()
                    adapted[:, :3, :, :] = rgb_param
                    adapted[:, 3:4, :, :] = rgb_param.mean(dim=1, keepdim=True)
                    state_dict_spec[key] = adapted

        self.branch_spectral.load_state_dict(state_dict_spec)

    def forward(self, rgb, spectral):
        feat_rgb = self.branch_rgb(rgb)
        feat_spec = self.branch_spectral(spectral)

        fused = torch.cat([feat_rgb, feat_spec], dim=1)
        fused = self.fusion_conv(fused)

        embeddings = self.embedding_head(fused)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        output = self.decoder(fused)

        return output, embeddings


def combined_loss(y_pred, y_true, fn_1="dice", fn_2="bce"):
    fns = {
        "dice": smp.losses.DiceLoss(mode="binary"),
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "focal": smp.losses.FocalLoss(mode="binary"),
    }
    return 0.5 * fns[fn_1](y_pred, y_true) + 0.5 * fns[fn_2](y_pred, y_true)


def _sample_pixel_triplets(embeddings, masks, n_triplets=TRIPLET_SAMPLES):
    bsz, emb_dim, height, width = embeddings.shape
    emb_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, emb_dim)
    mask_flat = masks.reshape(-1) > 0.5

    pos_idx = torch.where(mask_flat)[0]
    neg_idx = torch.where(~mask_flat)[0]

    if pos_idx.numel() < 2 or neg_idx.numel() < 1:
        return None, None, None

    n = min(n_triplets, pos_idx.numel(), neg_idx.numel())

    anchor_idx = pos_idx[torch.randint(pos_idx.numel(), (n,), device=embeddings.device)]
    positive_idx = pos_idx[torch.randint(pos_idx.numel(), (n,), device=embeddings.device)]
    negative_idx = neg_idx[torch.randint(neg_idx.numel(), (n,), device=embeddings.device)]

    return emb_flat[anchor_idx], emb_flat[positive_idx], emb_flat[negative_idx]


def compute_triplet_loss(embeddings, masks, margin=TRIPLET_MARGIN, n_triplets=TRIPLET_SAMPLES):
    anchor, positive, negative = _sample_pixel_triplets(embeddings, masks, n_triplets=n_triplets)
    if anchor is None:
        return embeddings.new_tensor(0.0)
    return nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)


def load_cbers4_dataset(data_dir: str):
    band7_dir = os.path.join(data_dir, "bands/BAND7")
    if os.path.exists(band7_dir):
        red_dir = band7_dir
        nir_dir = os.path.join(data_dir, "bands/BAND8")
        blue_dir = os.path.join(data_dir, "bands/BAND5")
        green_dir = os.path.join(data_dir, "bands/BAND6")
        mask_dir = os.path.join(data_dir, "groundtruth")
    else:
        red_dir = os.path.join(data_dir, "red")
        green_dir = os.path.join(data_dir, "green")
        blue_dir = os.path.join(data_dir, "blue")
        nir_dir = os.path.join(data_dir, "nir")
        mask_dir = os.path.join(data_dir, "masks_ibge")

    ids = sorted([f for f in os.listdir(red_dir) if f.endswith(".tiff") or f.endswith(".tif")])

    return (
        [os.path.join(red_dir, f) for f in ids],
        [os.path.join(nir_dir, f) for f in ids],
        [os.path.join(blue_dir, f) for f in ids],
        [os.path.join(green_dir, f) for f in ids],
        [os.path.join(mask_dir, f) for f in ids],
    )


def create_dataloaders(train_paths, val_paths, img_size=256, batch_size=8, crop_factor=1):
    train_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        ToTensorV2(),
    ])

    train_ds = CBERS4MUXSiamese2BranchDataset(
        red_image_paths=train_paths["red"],
        nir_image_paths=train_paths["nir"],
        blue_image_paths=train_paths["blue"],
        green_image_paths=train_paths["green"],
        mask_paths=train_paths["masks"],
        crop_factor=crop_factor,
        transform=train_transform,
        indices_to_add=["NDVI", "NDWI", "GNDVI"],
    )

    val_ds = CBERS4MUXSiamese2BranchDataset(
        red_image_paths=val_paths["red"],
        nir_image_paths=val_paths["nir"],
        blue_image_paths=val_paths["blue"],
        green_image_paths=val_paths["green"],
        mask_paths=val_paths["masks"],
        crop_factor=crop_factor,
        transform=val_transform,
        indices_to_add=["NDVI", "NDWI", "GNDVI"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader


def save_predictions(model, loader, device, num_examples=5, results_dir="./results"):
    print("Salvando exemplos de previsoes...")
    model.eval()

    if len(loader) == 0:
        print("Loader de validacao vazio. Pulando.")
        return

    rgb, spectral, masks = next(iter(loader))
    rgb, spectral, masks = rgb.to(device), spectral.to(device), masks.to(device)

    with torch.no_grad():
        outputs, _embeddings = model(rgb, spectral)
        preds = torch.sigmoid(outputs)

    rgb_cpu = rgb.cpu()
    masks_cpu = masks.cpu().numpy()
    preds_cpu = preds.cpu().numpy()

    for i in range(min(num_examples, len(rgb_cpu))):
        img_rgb = rgb_cpu[i].permute(1, 2, 0).numpy()
        img_rgb = (img_rgb - img_rgb.min()) / max(np.ptp(img_rgb), 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.clip(img_rgb, 0, 1))
        axes[0].set_title("Imagem RGB")
        axes[0].axis("off")

        axes[1].imshow(masks_cpu[i].squeeze(), cmap="gray")
        axes[1].set_title("Mascara Real")
        axes[1].axis("off")

        axes[2].imshow(preds_cpu[i].squeeze(), cmap="gray")
        axes[2].set_title("Predicao Siamesa 2B")
        axes[2].axis("off")

        pred_path = os.path.join(results_dir, f"prediction_example_{i + 1}.png")
        plt.savefig(pred_path)
        plt.close(fig)


def train_simple(model, device, train_loader, val_loader, loss_fn,
                 num_epochs, results_dir, backbone):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = {
        "train_loss": [], "train_seg_loss": [], "train_triplet_loss": [],
        "val_loss": [], "val_seg_loss": [], "val_triplet_loss": [],
        "val_iou": [],
    }

    best_iou = 0.0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        if train_loader.batch_size == 1:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_triplet_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{num_epochs} [Train]")

        for rgb, spectral, masks in pbar:
            rgb = rgb.to(device)
            spectral = spectral.to(device)
            masks = masks.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, embeddings = model(rgb, spectral)
                seg_loss = loss_fn(outputs, masks)
                triplet_loss = compute_triplet_loss(embeddings, masks)
                loss = seg_loss + TRIPLET_WEIGHT * triplet_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_triplet_loss += triplet_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg": f"{seg_loss.item():.4f}",
                "trip": f"{triplet_loss.item():.4f}",
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_seg = train_seg_loss / len(train_loader)
        avg_train_trip = train_triplet_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_seg_loss"].append(avg_train_seg)
        history["train_triplet_loss"].append(avg_train_trip)

        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_triplet_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Ep {epoch + 1}/{num_epochs} [Val]")
            for rgb, spectral, masks in vbar:
                rgb = rgb.to(device)
                spectral = spectral.to(device)
                masks = masks.to(device)

                outputs, embeddings = model(rgb, spectral)
                seg_loss = loss_fn(outputs, masks)
                triplet_loss = compute_triplet_loss(embeddings, masks)
                loss = seg_loss + TRIPLET_WEIGHT * triplet_loss

                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_triplet_loss += triplet_loss.item()

                preds = torch.sigmoid(outputs) > 0.5

                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds, masks.long(), mode="binary", threshold=0.5
                )
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                val_iou += iou.item()

                vbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "seg": f"{seg_loss.item():.4f}",
                    "iou": f"{iou.item():.4f}",
                })

        avg_val_loss = val_loss / len(val_loader)
        avg_val_seg = val_seg_loss / len(val_loader)
        avg_val_trip = val_triplet_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_seg_loss"].append(avg_val_seg)
        history["val_triplet_loss"].append(avg_val_trip)
        history["val_iou"].append(avg_val_iou)

        print(
            f"\nEp {epoch + 1}/{num_epochs} -> "
            f"Train L: {avg_train_loss:.4f} (Seg: {avg_train_seg:.4f}, Trip: {avg_train_trip:.4f}) | "
            f"Val L: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f}"
        )

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_epoch = epoch + 1
            os.makedirs(results_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            print(f"  -> Novo melhor modelo! IoU: {best_iou:.4f}")

        scheduler.step(avg_val_loss)

        if epoch > 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss")
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(history["val_iou"], label="Val IoU", color="green")
            plt.title("IoU")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "learning_curve.png"))
            plt.close()

    print(f"\nConcluido. Melhor IoU: {best_iou:.4f} (epoca {best_epoch})")

    best_model_path = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        save_predictions(model, val_loader, device, num_examples=5, results_dir=results_dir)

    history_path = os.path.join(results_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return {
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teste rapido Siamese 2 branches (RGB + NIR/Indices)")
    parser.add_argument("--data_dir", type=str, default="../datasets/doce_256",
                        help="Diretório(s) do dataset, separados por vírgula")
    parser.add_argument("--backbone", type=str, default="resnet152",
                        help="Any backbone supported by segmentation_models_pytorch (e.g. resnet152, efficientnet-b5, mit_b5)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="./test_results")
    parser.add_argument("--img_size", type=int, default=512, help="Tamanho da imagem redimensionada")
    parser.add_argument("--crop_factor", type=int, default=1,
                        help="Fator de crop em grade NxN. Ex: 1 para original, 2 para 2x2 = 4 crops, etc.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Epocas: {args.epochs}")
    print(f"Batch: {args.batch_size}")
    print(f"Val split: {args.val_split}")
    print(f"Img size: {args.img_size}")
    print(f"Crop factor: {args.crop_factor} ({args.crop_factor * args.crop_factor} tiles por imagem)")

    data_dirs = [d.strip() for d in args.data_dir.split(",")]
    red, nir, blue, green, masks = [], [], [], [], []
    for d in data_dirs:
        r, n, b, g, m = load_cbers4_dataset(d)
        red.extend(r); nir.extend(n); blue.extend(b)
        green.extend(g); masks.extend(m)
    print(f"Datasets: {data_dirs}")
    print(f"Total de amostras: {len(red)}")

    indices = list(range(len(red)))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_split, random_state=42, shuffle=True
    )

    train_paths = {
        "red": [red[i] for i in train_idx],
        "nir": [nir[i] for i in train_idx],
        "blue": [blue[i] for i in train_idx],
        "green": [green[i] for i in train_idx],
        "masks": [masks[i] for i in train_idx],
    }
    val_paths = {
        "red": [red[i] for i in val_idx],
        "nir": [nir[i] for i in val_idx],
        "blue": [blue[i] for i in val_idx],
        "green": [green[i] for i in val_idx],
        "masks": [masks[i] for i in val_idx],
    }

    print(f"Treino: {len(train_idx)} amostras (x{args.crop_factor * args.crop_factor} crops = {len(train_idx) * args.crop_factor * args.crop_factor} tiles) | Val: {len(val_idx)} amostras")

    train_loader, val_loader = create_dataloaders(
        train_paths, val_paths, img_size=args.img_size, batch_size=args.batch_size,
        crop_factor=args.crop_factor,
    )

    model = Siamese2BranchNet(encoder_name=args.backbone)

    run_name = f"{args.backbone}_siamese_2branch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)

    metrics = train_simple(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=combined_loss,
        num_epochs=args.epochs,
        results_dir=results_dir,
        backbone=args.backbone,
    )

    print("\n" + "=" * 50)
    print("RESULTADOS:")
    print(f"  Melhor IoU:  {metrics['best_iou']:.4f}")
    print(f"  Melhor epoca: {metrics['best_epoch']}")
    print(f"  Train loss final: {metrics['final_train_loss']:.4f}")
    print(f"  Val loss final:   {metrics['final_val_loss']:.4f}")
    print(f"Resultados salvos em: {results_dir}")
    print("=" * 50)
