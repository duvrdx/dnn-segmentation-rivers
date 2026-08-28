import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

from datasets import CBERS4MUXDataset

LEARNING_RATE = 1e-4
NUM_EPOCHS = 150
TRIPLET_WEIGHT = 0.1
TRIPLET_MARGIN = 0.2
TRIPLET_SAMPLES = 2048
EMBEDDING_DIM = 32


class CBERS4MUXSiamese2BranchDataset(Dataset):
    """Retorna RGB e NIR+indices separados, sem crop (imagem inteira 256x256)."""

    def __init__(self, red_image_paths, green_image_paths, blue_image_paths,
                 nir_image_paths, mask_paths, indices_to_add=None,
                 transform=None, mask_invert='auto'):
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
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]
        rgb = image[:3, :, :]
        spectral = image[3:, :, :]
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

        if isinstance(feat_rgb, tuple):
            feat_rgb = feat_rgb[0]
        if isinstance(feat_spec, tuple):
            feat_spec = feat_spec[0]

        fused = torch.cat([feat_rgb, feat_spec], dim=1)
        fused = self.fusion_conv(fused)

        embeddings = self.embedding_head(fused)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        output = self.decoder(fused)

        return output, embeddings


def _normalize_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _otsu_threshold_from_prob_map(prob_map: np.ndarray) -> float:
    flat = np.clip(prob_map.ravel(), 0.0, 1.0)
    if flat.size == 0:
        return 0.5
    hist, bin_edges = np.histogram(flat, bins=256, range=(0.0, 1.0))
    if hist.max() == flat.size:
        return 0.5
    prob = hist.astype(np.float64) / flat.size
    omega = np.cumsum(prob)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    sigma_b2[(omega <= 0.0) | (omega >= 1.0)] = -1.0
    threshold = float(bin_centers[np.argmax(sigma_b2)])
    return float(np.clip(threshold, 0.05, 0.95))


def _binarize_with_otsu(outputs: torch.Tensor):
    outputs_np = outputs.detach().cpu().numpy()
    preds_np = np.zeros_like(outputs_np, dtype=np.uint8)
    thresholds = []
    for i in range(outputs_np.shape[0]):
        prob_map = outputs_np[i, 0]
        threshold = _otsu_threshold_from_prob_map(prob_map)
        thresholds.append(threshold)
        preds_np[i, 0] = (prob_map >= threshold).astype(np.uint8)
    preds = torch.from_numpy(preds_np).to(outputs.device)
    return preds, thresholds


def load_cbers4_dataset(data_dir: str):
    red_dir = os.path.join(data_dir, 'bands/BAND7')
    green_dir = os.path.join(data_dir, 'bands/BAND6')
    blue_dir = os.path.join(data_dir, 'bands/BAND5')
    nir_dir = os.path.join(data_dir, 'bands/BAND8')
    mask_dir = os.path.join(data_dir, 'groundtruth')

    ids = sorted([f for f in os.listdir(red_dir)
                 if f.endswith(".tiff") or f.endswith(".tif")])

    return (
        [os.path.join(red_dir, f) for f in ids],
        [os.path.join(nir_dir, f) for f in ids],
        [os.path.join(blue_dir, f) for f in ids],
        [os.path.join(green_dir, f) for f in ids],
        [os.path.join(mask_dir, f) for f in ids],
    )


def create_dataloaders(train_paths, val_paths, img_size=128, batch_size=8):
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
        transform=train_transform,
        indices_to_add=["NDVI", "NDWI", "GNDVI"],
    )

    val_ds = CBERS4MUXSiamese2BranchDataset(
        red_image_paths=val_paths["red"],
        nir_image_paths=val_paths["nir"],
        blue_image_paths=val_paths["blue"],
        green_image_paths=val_paths["green"],
        mask_paths=val_paths["masks"],
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
    rgb, spectral, masks = rgb.to(
        device), spectral.to(device), masks.to(device)

    with torch.no_grad():
        outputs, _embeddings = model(rgb, spectral)
        preds, otsu_thresholds = _binarize_with_otsu(torch.sigmoid(outputs))

    print(f"Threshold Otsu medio do batch: {np.mean(otsu_thresholds):.4f}")

    rgb_cpu = rgb.cpu()
    masks_cpu = masks.cpu().numpy()
    preds_cpu = preds.cpu().numpy()

    for i in range(min(num_examples, len(rgb_cpu))):
        img_rgb = rgb_cpu[i].permute(1, 2, 0).numpy()
        img_rgb = _normalize_2d(img_rgb)

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

        pred_path = os.path.join(
            results_dir, f"prediction_example_{i + 1}.png")
        plt.savefig(pred_path)
        plt.close(fig)


def _sample_pixel_triplets(embeddings, masks, n_triplets=TRIPLET_SAMPLES):
    bsz, emb_dim, height, width = embeddings.shape
    emb_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, emb_dim)
    mask_flat = masks.reshape(-1) > 0.5

    pos_idx = torch.where(mask_flat)[0]
    neg_idx = torch.where(~mask_flat)[0]

    if pos_idx.numel() < 2 or neg_idx.numel() < 1:
        return None, None, None

    n = min(n_triplets, pos_idx.numel(), neg_idx.numel())

    anchor_idx = pos_idx[torch.randint(
        pos_idx.numel(), (n,), device=embeddings.device)]
    positive_idx = pos_idx[torch.randint(
        pos_idx.numel(), (n,), device=embeddings.device)]
    negative_idx = neg_idx[torch.randint(
        neg_idx.numel(), (n,), device=embeddings.device)]

    return emb_flat[anchor_idx], emb_flat[positive_idx], emb_flat[negative_idx]


def compute_triplet_loss(embeddings, masks, margin=TRIPLET_MARGIN, n_triplets=TRIPLET_SAMPLES):
    anchor, positive, negative = _sample_pixel_triplets(
        embeddings, masks, n_triplets=n_triplets)
    if anchor is None:
        return embeddings.new_tensor(0.0)
    return nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)


def combined_loss(y_pred, y_true, fn_1="dice", fn_2="bce"):
    fns = {
        "dice": smp.losses.DiceLoss(mode="binary"),
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "focal": smp.losses.FocalLoss(mode="binary"),
    }
    return 0.5 * fns[fn_1](y_pred, y_true) + 0.5 * fns[fn_2](y_pred, y_true)


def execute_siamese_2branch_train(
    model, device, train_loader, val_loader, loss_fn, num_epochs, results_dir
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
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
        train_loss = 0.0
        train_seg_loss = 0.0
        train_triplet_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")

        for rgb, spectral, masks in pbar:
            rgb, spectral, masks = rgb.to(
                device), spectral.to(device), masks.to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
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
            vbar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")

            for rgb, spectral, masks in vbar:
                rgb, spectral, masks = rgb.to(
                    device), spectral.to(device), masks.to(device)

                outputs, embeddings = model(rgb, spectral)
                seg_loss = loss_fn(outputs, masks)
                triplet_loss = compute_triplet_loss(embeddings, masks)
                loss = seg_loss + TRIPLET_WEIGHT * triplet_loss

                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_triplet_loss += triplet_loss.item()

                preds, _ = _binarize_with_otsu(torch.sigmoid(outputs))

                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds.float(), masks.long(), mode="binary", threshold=0.5
                )
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                val_iou += iou.item()

                vbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "seg": f"{seg_loss.item():.4f}",
                    "trip": f"{triplet_loss.item():.4f}",
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
            f"\nEpoch {epoch + 1}/{num_epochs} -> "
            f"Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg:.4f}, Trip: {avg_train_trip:.4f}), "
            f"Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg:.4f}, Trip: {avg_val_trip:.4f}), "
            f"Val IoU: {avg_val_iou:.4f}"
        )

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_epoch = epoch + 1
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"Novo melhor modelo salvo em {model_path} com IoU: {best_iou:.4f}")

        scheduler.step(avg_val_loss)

        if epoch > 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.title("Curva de Perda")
            plt.xlabel("Epocas")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(history["val_iou"], label="Validation IoU", color="green")
            plt.title("Curva de IoU")
            plt.xlabel("Epocas")
            plt.ylabel("IoU")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            learning_curve_path = os.path.join(
                results_dir, "learning_curve.png")
            plt.savefig(learning_curve_path)
            plt.close()

    print(f"\nTreinamento concluido. Melhor IoU: {best_iou:.4f}")

    best_model_path = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        save_predictions(model, val_loader, device,
                         num_examples=6, results_dir=results_dir)

    history_path = os.path.join(results_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return {
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_train_seg_loss": avg_train_seg,
        "final_train_triplet_loss": avg_train_trip,
        "final_val_seg_loss": avg_val_seg,
        "final_val_triplet_loss": avg_val_trip,
    }


# --- Configuração dos Experimentos ---
if __name__ == "__main__":
    trains = {
        'complete': {
            'train_dirs': ['../Datasets/itapemirim_river', '../Datasets/doce_river'],
            'test_dirs': None
        }
    }

    backbones = [
        "resnet34",
        # "timm-efficientnet-b7",
        # "mit_b5",
    ]

    all_results_summary = []

    for train_num in range(1):
        for train_name, config in trains.items():
            for backbone in backbones:
                print(
                    f"\n{'=' * 80}\nExperimento: {train_name} | Backbone: {backbone} [Siamese 2-Branch]\n{'=' * 80}"
                )

                train_red, train_nir, train_blue, train_green, train_masks = [], [], [], [], []
                for data_dir in config["train_dirs"]:
                    r, n, b, g, m = load_cbers4_dataset(data_dir)
                    train_red.extend(r)
                    train_nir.extend(n)
                    train_blue.extend(b)
                    train_green.extend(g)
                    train_masks.extend(m)

                if not config["test_dirs"]:
                    (
                        train_red_paths, val_red_paths,
                        train_nir_paths, val_nir_paths,
                        train_blue_paths, val_blue_paths,
                        train_green_paths, val_green_paths,
                        train_mask_paths, val_mask_paths,
                    ) = train_test_split(
                        train_red, train_nir, train_blue, train_green, train_masks,
                        test_size=0.2, random_state=42,
                    )
                else:
                    train_red_paths, train_nir_paths = train_red, train_nir
                    train_blue_paths, train_green_paths = train_blue, train_green
                    train_mask_paths = train_masks

                    val_red_paths, val_nir_paths = [], []
                    val_blue_paths, val_green_paths = [], []
                    val_mask_paths = []
                    for data_dir in config["test_dirs"]:
                        r, n, b, g, m = load_cbers4_dataset(data_dir)
                        val_red_paths.extend(r)
                        val_nir_paths.extend(n)
                        val_blue_paths.extend(b)
                        val_green_paths.extend(g)
                        val_mask_paths.extend(m)

                train_paths_dict = {
                    "red": train_red_paths, "nir": train_nir_paths,
                    "blue": train_blue_paths, "green": train_green_paths,
                    "masks": train_mask_paths,
                }
                val_paths_dict = {
                    "red": val_red_paths, "nir": val_nir_paths,
                    "blue": val_blue_paths, "green": val_green_paths,
                    "masks": val_mask_paths,
                }

                train_loader, val_loader = create_dataloaders(
                    train_paths=train_paths_dict, val_paths=val_paths_dict,
                    img_size=512, batch_size=4,
                )

                model = Siamese2BranchNet(encoder_name=backbone)

                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                results_dir = (
                    f"./experiments/stats_siamese_2branch/{train_num}/"
                    f"results_{train_name}_{backbone}_siamese_2branch"
                )

                final_metrics = execute_siamese_2branch_train(
                    model=model, device=device,
                    train_loader=train_loader, val_loader=val_loader,
                    loss_fn=combined_loss, num_epochs=NUM_EPOCHS,
                    results_dir=results_dir,
                )

                summary_line = (
                    f"Experimento: {train_name:<30} | Backbone: {backbone:<18}\n"
                    f"  -> Melhor IoU de Validacao: {final_metrics['best_iou']:.4f} "
                    f"(na epoca {final_metrics['best_epoch']})\n"
                    f"  -> Perda Final (Treino/Val): {final_metrics['final_train_loss']:.4f} / "
                    f"{final_metrics['final_val_loss']:.4f}\n"
                    f"  -> Perda Final Seg (Treino/Val): {final_metrics['final_train_seg_loss']:.4f} / "
                    f"{final_metrics['final_val_seg_loss']:.4f}\n"
                    f"  -> Perda Final Triplet (Treino/Val): {final_metrics['final_train_triplet_loss']:.4f} / "
                    f"{final_metrics['final_val_triplet_loss']:.4f}\n"
                )
                all_results_summary.append(summary_line)
                print(summary_line)

    summary_filepath = "final_results_summary_siamese_2branch.txt"
    print(f"\nSalvando resumo final em: {summary_filepath}")

    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write("--- RESUMO FINAL DOS EXPERIMENTOS (SIAMESE 2-BRANCH) ---\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 40 + "\n\n")
        for summary in all_results_summary:
            f.write(summary)
            f.write("-" * 60 + "\n")

    print("Resumo salvo com sucesso.")
