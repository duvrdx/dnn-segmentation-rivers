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
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import CBERS4MUXDatasetWHidrography
from utils import (
    create_kfold_splits,
    compute_fold_statistics,
    save_fold_results,
    save_holdout_results,
    format_summary_line,
)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
TRIPLET_WEIGHT = 0.5
TRIPLET_MARGIN = 0.5
TRIPLET_SAMPLES = 2048
EMBEDDING_DIM = 32
HYDRO_WEIGHT = 0.5


def _find_hydro_dir(data_dir: str) -> str:
    candidates = ["hidrography", "hydrography", "hidrografia", "hydro"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Nenhuma pasta de hidrografia encontrada em {data_dir}. "
        f"Tentativas: {candidates}"
    )


class CBERS4MUXMultiBranchDataset(Dataset):
    def __init__(self, red_image_paths, green_image_paths, blue_image_paths,
                 nir_image_paths, mask_paths, hidrography_paths,
                 indices_to_add=None, transform=None, mask_invert="auto"):
        self.base = CBERS4MUXDatasetWHidrography(
            red_image_paths=red_image_paths,
            green_image_paths=green_image_paths,
            blue_image_paths=blue_image_paths,
            nir_image_paths=nir_image_paths,
            mask_paths=mask_paths,
            hidrography_paths=hidrography_paths,
            indices_to_add=indices_to_add,
            transform=transform,
            mask_invert=mask_invert,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]
        # image: [B, G, R, NIR, HYDRO, NDVI, NDWI, GNDVI]
        rgb = image[:3, :, :]
        hydro = image[4:5, :, :]
        spectral = torch.cat([image[3:4, :, :], image[5:, :, :]], dim=0)
        return rgb, spectral, hydro, mask


class SiameseMultiBranchNet(nn.Module):
    def __init__(self, encoder_name: str = "timm-efficientnet-b7"):
        super().__init__()

        self.branch_rgb = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=256,
        )

        self.branch_spectral = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=4,
            classes=256,
        )
        self._init_branch_from_rgb(self.branch_spectral, 4)

        self.branch_hydro = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=1,
            classes=256,
        )
        self._init_branch_from_rgb(self.branch_hydro, 1)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
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

    def _init_branch_from_rgb(self, target_branch, target_in_channels):
        sd_rgb = self.branch_rgb.state_dict()
        sd_target = target_branch.state_dict()

        for key in sd_target.keys():
            if key in sd_rgb:
                rgb_param = sd_rgb[key]
                target_param = sd_target[key]

                if rgb_param.shape == target_param.shape:
                    sd_target[key] = rgb_param
                    continue

                if (len(rgb_param.shape) == 4 and len(target_param.shape) == 4
                        and rgb_param.shape[0] == target_param.shape[0]
                        and rgb_param.shape[2:] == target_param.shape[2:]
                        and rgb_param.shape[1] == 3):
                    adapted = target_param.clone()
                    n_copy = min(3, target_in_channels)
                    adapted[:, :n_copy, :, :] = rgb_param[:, :n_copy, :, :]
                    if target_in_channels > 3:
                        adapted[:, 3:, :, :] = rgb_param.mean(
                            dim=1, keepdim=True)
                    elif target_in_channels == 1:
                        adapted[:, 0:1, :, :] = rgb_param.mean(
                            dim=1, keepdim=True)
                    sd_target[key] = adapted

        target_branch.load_state_dict(sd_target)

    def forward(self, rgb, spectral, hydro):
        feat_rgb = self.branch_rgb(rgb)
        feat_spec = self.branch_spectral(spectral)
        feat_hydro = self.branch_hydro(hydro)

        fused = torch.cat([feat_rgb, feat_spec, feat_hydro], dim=1)
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


def otsu_threshold(prob_map: np.ndarray) -> float:
    prob_map = np.clip(prob_map, 0, 1)
    prob_map = (prob_map * 255).astype(np.uint8)
    hist = np.bincount(prob_map.ravel(), minlength=256)
    hist = hist / hist.sum()
    best_thresh = 0.5
    best_var = 0.0
    for t in range(1, 255):
        w0 = hist[:t].sum()
        w1 = hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.arange(t) @ hist[:t] / w0
        mu1 = np.arange(t, 256) @ hist[t:] / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t / 255.0
    return best_thresh


def _guided_binarize(outputs: torch.Tensor, hydro_maps: torch.Tensor, hydro_weight: float = HYDRO_WEIGHT):
    out_np = outputs.detach().cpu().numpy()
    hydro_np = hydro_maps.detach().cpu().numpy()
    preds_np = np.zeros_like(out_np, dtype=np.uint8)
    for i in range(out_np.shape[0]):
        prob = np.clip(out_np[i, 0], 0.0, 1.0)
        hydro = _normalize_2d(hydro_np[i, 0])
        fused_score = (1.0 - hydro_weight) * prob + hydro_weight * hydro
        thresh = otsu_threshold(fused_score)
        preds_np[i, 0] = (fused_score >= thresh).astype(np.uint8)
    return torch.from_numpy(preds_np).to(outputs.device)


def compute_wiou(preds: torch.Tensor, masks: torch.Tensor, alpha: float = 2.0):
    preds_np = preds.cpu().numpy()
    masks_np = masks.cpu().numpy()
    wiou_list = []
    for i in range(preds_np.shape[0]):
        gt = masks_np[i, 0].astype(np.float64)
        pred = preds_np[i, 0].astype(np.float64)
        dist_bg = ndimage.distance_transform_edt(1 - gt)
        dist_fg = ndimage.distance_transform_edt(gt)
        boundary_weight = 1.0 + alpha * np.exp(-0.5 * (dist_bg + dist_fg))
        intersection = np.sum((pred > 0.5) * (gt > 0.5) * boundary_weight)
        union = np.sum(((pred > 0.5) + (gt > 0.5) > 0) * boundary_weight)
        wiou_list.append(intersection / max(union, 1e-8))
    return float(np.mean(wiou_list))


def batch_hard_triplet_loss(embeddings, masks, margin=TRIPLET_MARGIN, n_samples=TRIPLET_SAMPLES):
    b, c, h, w = embeddings.shape
    emb_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, c)
    masks_resized = F.interpolate(masks, size=(h, w), mode='nearest')
    mask_flat = (masks_resized.reshape(-1) > 0.5)

    n_pos = mask_flat.sum().item()
    n_neg = (~mask_flat).sum().item()

    if n_pos < 2 or n_neg < 1:
        return embeddings.new_tensor(0.0)

    n_sample = min(n_samples, n_pos, n_neg, b * h * w)
    pos_idx = torch.where(mask_flat)[0]
    neg_idx = torch.where(~mask_flat)[0]

    perm = torch.randperm(pos_idx.numel(), device=embeddings.device)[:n_sample]
    anchor = emb_flat[pos_idx[perm]]
    pos = emb_flat[pos_idx[perm]]

    neg_perm = torch.randperm(
        neg_idx.numel(), device=embeddings.device)[:n_sample]
    neg = emb_flat[neg_idx[neg_perm]]

    dist_pos = torch.norm(anchor - pos, p=2, dim=1)
    dist_neg = torch.norm(anchor - neg, p=2, dim=1)

    loss = torch.clamp(dist_pos - dist_neg + margin, min=0)
    return loss.mean()


def combined_loss(y_pred, y_true, fn_1="dice", fn_2="bce"):
    fns = {
        "dice": smp.losses.DiceLoss(mode="binary"),
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "focal": smp.losses.FocalLoss(mode="binary"),
    }
    return 0.5 * fns[fn_1](y_pred, y_true) + 0.5 * fns[fn_2](y_pred, y_true)


def load_cbers4_dataset_with_hydro(data_dir: str):
    red_dir = os.path.join(data_dir, "red")
    green_dir = os.path.join(data_dir, "green")
    blue_dir = os.path.join(data_dir, "blue")
    nir_dir = os.path.join(data_dir, "nir")
    mask_dir = os.path.join(data_dir, "masks")
    hydro_dir = _find_hydro_dir(data_dir)

    ids = sorted([f for f in os.listdir(red_dir)
                 if f.endswith(".tiff") or f.endswith(".tif")])

    return (
        [os.path.join(red_dir, f) for f in ids],
        [os.path.join(nir_dir, f) for f in ids],
        [os.path.join(blue_dir, f) for f in ids],
        [os.path.join(green_dir, f) for f in ids],
        [os.path.join(mask_dir, f) for f in ids],
        [os.path.join(hydro_dir, f) for f in ids],
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

    train_ds = CBERS4MUXMultiBranchDataset(
        red_image_paths=train_paths["red"],
        nir_image_paths=train_paths["nir"],
        blue_image_paths=train_paths["blue"],
        green_image_paths=train_paths["green"],
        mask_paths=train_paths["masks"],
        hidrography_paths=train_paths["hydro"],
        transform=train_transform,
        indices_to_add=["NDVI", "NDWI", "GNDVI"],
    )

    val_ds = CBERS4MUXMultiBranchDataset(
        red_image_paths=val_paths["red"],
        nir_image_paths=val_paths["nir"],
        blue_image_paths=val_paths["blue"],
        green_image_paths=val_paths["green"],
        mask_paths=val_paths["masks"],
        hidrography_paths=val_paths["hydro"],
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
    rgb, spectral, hydro, masks = next(iter(loader))
    rgb, spectral, hydro, masks = (
        rgb.to(device), spectral.to(device), hydro.to(device), masks.to(device)
    )
    with torch.no_grad():
        outputs, _embeddings = model(rgb, spectral, hydro)
        preds = _guided_binarize(torch.sigmoid(
            outputs), hydro, hydro_weight=HYDRO_WEIGHT)
    rgb_cpu = rgb.cpu()
    masks_cpu = masks.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    hydro_cpu = hydro.cpu().numpy()
    for i in range(min(num_examples, len(rgb_cpu))):
        img_rgb = rgb_cpu[i].permute(1, 2, 0).numpy()
        img_rgb = _normalize_2d(img_rgb)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(np.clip(img_rgb, 0, 1))
        axes[0].set_title("RGB")
        axes[0].axis("off")
        axes[1].imshow(hydro_cpu[i].squeeze(), cmap="Blues")
        axes[1].set_title("Hidrografia")
        axes[1].axis("off")
        axes[2].imshow(masks_cpu[i].squeeze(), cmap="gray")
        axes[2].set_title("Mascara")
        axes[2].axis("off")
        axes[3].imshow(preds_cpu[i].squeeze(), cmap="gray")
        axes[3].set_title("Predicao")
        axes[3].axis("off")
        plt.savefig(os.path.join(
            results_dir, f"prediction_example_{i + 1}.png"))
        plt.close(fig)


def train_one_fold(
    model, device, train_loader, val_loader, loss_fn,
    num_epochs, results_dir, fold_idx=None,
    hydro_weight=HYDRO_WEIGHT,
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
        "val_iou": [], "val_wiou": [],
    }

    best_iou = 0.0
    best_wiou = 0.0
    best_epoch = -1
    fold_tag = f"Fold {fold_idx}" if fold_idx is not None else "Hold-out"

    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_triplet_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"{fold_tag} Ep {epoch + 1}/{num_epochs} [Train]")

        for rgb, spectral, hydro_maps, masks in pbar:
            rgb = rgb.to(device)
            spectral = spectral.to(device)
            hydro_maps = hydro_maps.to(device)
            masks = masks.to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs, embeddings = model(rgb, spectral, hydro_maps)
                seg_loss = loss_fn(outputs, masks)
                triplet_loss = batch_hard_triplet_loss(embeddings, masks)
                loss = seg_loss + TRIPLET_WEIGHT * triplet_loss

            if not torch.isfinite(loss):
                print(
                    f"  WARNING: NaN/Inf loss detected (seg={seg_loss.item():.4f}, trip={triplet_loss.item():.4f}), skipping batch")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        val_wiou = 0.0

        with torch.no_grad():
            vbar = tqdm(
                val_loader, desc=f"{fold_tag} Ep {epoch + 1}/{num_epochs} [Val]")
            for rgb, spectral, hydro_maps, masks in vbar:
                rgb = rgb.to(device)
                spectral = spectral.to(device)
                hydro_maps = hydro_maps.to(device)
                masks = masks.to(device)

                outputs, embeddings = model(rgb, spectral, hydro_maps)
                seg_loss = loss_fn(outputs, masks)
                triplet_loss = batch_hard_triplet_loss(embeddings, masks)
                loss = seg_loss + TRIPLET_WEIGHT * triplet_loss

                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_triplet_loss += triplet_loss.item()

                preds = _guided_binarize(outputs, hydro_maps, hydro_weight)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds, masks.long(), mode="binary", threshold=0.5
                )
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                val_iou += iou.item()

                wiou = compute_wiou(preds, masks)
                val_wiou += wiou

                vbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "seg": f"{seg_loss.item():.4f}",
                    "iou": f"{iou.item():.4f}",
                })

        avg_val_loss = val_loss / len(val_loader)
        avg_val_seg = val_seg_loss / len(val_loader)
        avg_val_trip = val_triplet_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_wiou = val_wiou / len(val_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_seg_loss"].append(avg_val_seg)
        history["val_triplet_loss"].append(avg_val_trip)
        history["val_iou"].append(avg_val_iou)
        history["val_wiou"].append(avg_val_wiou)

        print(
            f"\n{fold_tag} Epoch {epoch + 1}/{num_epochs} -> "
            f"Train L: {avg_train_loss:.4f} (Seg: {avg_train_seg:.4f}, Trip: {avg_train_trip:.4f}) | "
            f"Val L: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f} | WIoU: {avg_val_wiou:.4f}"
        )

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_wiou = avg_val_wiou
            best_epoch = epoch + 1
            os.makedirs(results_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                results_dir, "best_model.pth"))
            print(
                f"  -> Novo melhor modelo! IoU: {best_iou:.4f}, WIoU: {best_wiou:.4f}")

        scheduler.step(avg_val_loss)

        if epoch > 0:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss")
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(history["val_iou"], label="Val IoU", color="green")
            plt.title("IoU")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(history["val_wiou"], label="Val WIoU", color="orange")
            plt.title("WIoU (boundary-aware)")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "learning_curve.png"))
            plt.close()

    print(f"\n{fold_tag} concluido. Melhor IoU: {best_iou:.4f} (epoca {best_epoch}), WIoU: {best_wiou:.4f}")

    best_model_path = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        save_predictions(model, val_loader, device,
                         num_examples=5, results_dir=results_dir)

    history_path = os.path.join(results_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return {
        "best_iou": best_iou,
        "best_wiou": best_wiou,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }


def run_kfold_experiment(config, backbone, n_splits=5, results_root="./experiments/stats_siamese_multibranch"):
    train_red, train_nir, train_blue, train_green, train_masks, train_hydro = [], [], [], [], [], []
    for data_dir in config["train_dirs"]:
        r, n, b, g, m, h = load_cbers4_dataset_with_hydro(data_dir)
        train_red.extend(r)
        train_nir.extend(n)
        train_blue.extend(b)
        train_green.extend(g)
        train_masks.extend(m)
        train_hydro.extend(h)

    all_paths = {
        "red": train_red, "nir": train_nir,
        "blue": train_blue, "green": train_green,
        "masks": train_masks, "hydro": train_hydro,
    }

    splits = create_kfold_splits(all_paths, n_splits=n_splits)
    fold_metrics = []

    for fold_idx, (train_paths, val_paths) in enumerate(splits):
        print(f"\n{'=' * 60}\nFold {fold_idx + 1}/{n_splits}\n{'=' * 60}")

        train_loader, val_loader = create_dataloaders(
            train_paths, val_paths, img_size=128, batch_size=8
        )

        model = SiameseMultiBranchNet(encoder_name=backbone)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fold_dir = os.path.join(
            results_root, config["name"], backbone, f"fold_{fold_idx}"
        )

        metrics = train_one_fold(
            model=model, device=device,
            train_loader=train_loader, val_loader=val_loader,
            loss_fn=combined_loss, num_epochs=NUM_EPOCHS,
            results_dir=fold_dir, fold_idx=fold_idx,
        )
        fold_metrics.append(metrics)

    aggregated = compute_fold_statistics(fold_metrics)
    save_fold_results(
        os.path.join(results_root, config["name"], backbone),
        config["name"], backbone, fold_metrics, aggregated,
    )

    return fold_metrics, aggregated


def run_holdout_experiment(config, backbone, results_root="./experiments/stats_siamese_multibranch"):
    train_red, train_nir, train_blue, train_green, train_masks, train_hydro = [], [], [], [], [], []
    for data_dir in config["train_dirs"]:
        r, n, b, g, m, h = load_cbers4_dataset_with_hydro(data_dir)
        train_red.extend(r)
        train_nir.extend(n)
        train_blue.extend(b)
        train_green.extend(g)
        train_masks.extend(m)
        train_hydro.extend(h)

    val_red, val_nir, val_blue, val_green, val_masks, val_hydro = [], [], [], [], [], []
    for data_dir in config["test_dirs"]:
        r, n, b, g, m, h = load_cbers4_dataset_with_hydro(data_dir)
        val_red.extend(r)
        val_nir.extend(n)
        val_blue.extend(b)
        val_green.extend(g)
        val_masks.extend(m)
        val_hydro.extend(h)

    train_paths = {
        "red": train_red, "nir": train_nir,
        "blue": train_blue, "green": train_green,
        "masks": train_masks, "hydro": train_hydro,
    }
    val_paths = {
        "red": val_red, "nir": val_nir,
        "blue": val_blue, "green": val_green,
        "masks": val_masks, "hydro": val_hydro,
    }

    train_loader, val_loader = create_dataloaders(
        train_paths, val_paths, img_size=128, batch_size=8
    )

    model = SiameseMultiBranchNet(encoder_name=backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(
        results_root, config["name"], backbone
    )

    metrics = train_one_fold(
        model=model, device=device,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=combined_loss, num_epochs=NUM_EPOCHS,
        results_dir=results_dir, fold_idx=None,
    )

    save_holdout_results(results_dir, config["name"], backbone, metrics)

    return metrics


experiments = {
    # "doce": {
    #     "name": "doce",
    #     "type": "kfold",
    #     "train_dirs": ["../datasets/doce_cbers_mux"],
    # },
    # "itapemirim": {
    #     "name": "itapemirim",
    #     "type": "kfold",
    #     "train_dirs": ["../datasets/itapemirim_cbers_mux"],
    # },
    "itapemirim_doce": {
        "name": "itapemirim_doce",
        "type": "kfold",
        "train_dirs": ["../datasets/itapemirim_cbers_mux", "../datasets/doce_cbers_mux"],
    },
    # "train_doce_test_itapemirim": {
    #     "name": "train_doce_test_itapemirim",
    #     "type": "holdout",
    #     "train_dirs": ["../datasets/doce_cbers_mux"],
    #     "test_dirs": ["../datasets/itapemirim_cbers_mux"],
    # },
    # "train_itapemirim_test_doce": {
    #     "name": "train_itapemirim_test_doce",
    #     "type": "holdout",
    #     "train_dirs": ["../datasets/itapemirim_cbers_mux"],
    #     "test_dirs": ["../datasets/doce_cbers_mux"],
    # },
}

backbones = [
    # "timm-efficientnet-b7",
    # "mit_b5",
    "vgg11",
]


def run_simple_experiment(config, backbone, test_size=0.2, results_root="./experiments/stats_siamese_multibranch"):
    from sklearn.model_selection import train_test_split

    train_red, train_nir, train_blue, train_green, train_masks, train_hydro = [], [], [], [], [], []
    for data_dir in config["train_dirs"]:
        r, n, b, g, m, h = load_cbers4_dataset_with_hydro(data_dir)
        train_red.extend(r)
        train_nir.extend(n)
        train_blue.extend(b)
        train_green.extend(g)
        train_masks.extend(m)
        train_hydro.extend(h)

    n = len(train_red)
    idx = list(range(n))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, random_state=42)

    train_paths = {k: [v[i] for i in train_idx] for k, v in
                   zip(["red", "nir", "blue", "green", "masks", "hydro"],
                       [train_red, train_nir, train_blue, train_green, train_masks, train_hydro])}
    val_paths = {k: [v[i] for i in val_idx] for k, v in
                 zip(["red", "nir", "blue", "green", "masks", "hydro"],
                     [train_red, train_nir, train_blue, train_green, train_masks, train_hydro])}

    train_loader, val_loader = create_dataloaders(
        train_paths, val_paths, img_size=128, batch_size=8)

    model = SiameseMultiBranchNet(encoder_name=backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(results_root, config["name"], backbone)
    metrics = train_one_fold(
        model=model, device=device,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=combined_loss, num_epochs=NUM_EPOCHS,
        results_dir=results_dir, fold_idx=None,
    )

    save_holdout_results(results_dir, config["name"], backbone, metrics)
    return metrics


if __name__ == "__main__":
    results_root = "./experiments/stats_siamese_multibranch"
    test_config = experiments["itapemirim_doce"]
    test_backbone = "vgg11"

    print(f"\n{'#' * 80}")
    print(
        f"# EXPERIMENTO: {test_config['name']} | Backbone: {test_backbone} | Split: 80/20")
    print(f"{'#' * 80}")

    metrics = run_simple_experiment(
        test_config, test_backbone, test_size=0.2, results_root=results_root
    )

    summary_line = format_summary_line(
        test_config["name"], test_backbone, metrics, is_kfold=False
    )
    print(summary_line)

    summary_filepath = "final_results_summary_siamese_multibranch.txt"
    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write("--- SIAMESE MULTI-BRANCH (RGB + SPECTRAL + HYDRO) ---\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Config: {test_config['name']} | Backbone: {test_backbone}\n")
        f.write(
            f"Triplet weight: {TRIPLET_WEIGHT}, margin: {TRIPLET_MARGIN}\n")
        f.write(f"Split: 80/20 | Img size: 128x128\n\n")
        f.write(summary_line)

    print(f"Resultados salvos em: {summary_filepath}")
