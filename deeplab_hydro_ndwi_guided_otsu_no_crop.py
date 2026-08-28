import os
import pickle
from datetime import datetime

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import CBERS4MUXDatasetWHidrography
from scipy import ndimage

LEARNING_RATE = 1e-4
NUM_EPOCHS = 75
HYDRO_WEIGHT = 0.2
NDWI_WEIGHT = 0.4


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


def load_cbers4_dataset_with_hydro(data_dir: str):
    """Carrega caminhos de bandas, mascara e hidrografia de um unico diretorio."""
    red_dir = os.path.join(data_dir, "red")
    green_dir = os.path.join(data_dir, "green")
    blue_dir = os.path.join(data_dir, "blue")
    nir_dir = os.path.join(data_dir, "nir")
    mask_dir = os.path.join(data_dir, "masks_ibge")
    hydro_dir = _find_hydro_dir(data_dir)

    ids = sorted([f for f in os.listdir(red_dir) if f.endswith(".tiff") or f.endswith(".tif")])

    red_paths = [os.path.join(red_dir, f) for f in ids]
    nir_paths = [os.path.join(nir_dir, f) for f in ids]
    blue_paths = [os.path.join(blue_dir, f) for f in ids]
    green_paths = [os.path.join(green_dir, f) for f in ids]
    mask_paths = [os.path.join(mask_dir, f) for f in ids]
    hydro_paths = [os.path.join(hydro_dir, f) for f in ids]

    return red_paths, nir_paths, blue_paths, green_paths, mask_paths, hydro_paths


class CBERS4HydroGuidedDataset(Dataset):
    """
    Usa CBERS4MUXDatasetWHidrography para manter transformacoes alinhadas,
    mas remove o canal de hidrografia da entrada do modelo.
    """

    def __init__(
        self,
        red_image_paths,
        green_image_paths,
        blue_image_paths,
        nir_image_paths,
        mask_paths,
        hidrography_paths,
        indices_to_add=None,
        transform=None,
        mask_invert="auto",
    ):
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
        full_image, mask = self.base[idx]

        # Ordem dos canais vindos da base: [B, G, R, NIR, HYDRO, ...indices]
        hydro = full_image[4:5, :, :]
        image_wo_hydro = torch.cat([full_image[:4, :, :], full_image[5:, :, :]], dim=0)

        return image_wo_hydro, mask, hydro


def create_dataloaders(train_paths: dict, val_paths: dict, img_size: int = 256, batch_size: int = 8):
    train_transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            ToTensorV2(),
        ]
    )

    train_ds = CBERS4HydroGuidedDataset(
        red_image_paths=train_paths["red"],
        nir_image_paths=train_paths["nir"],
        blue_image_paths=train_paths["blue"],
        green_image_paths=train_paths["green"],
        mask_paths=train_paths["masks"],
        hidrography_paths=train_paths["hydro"],
        transform=train_transform,
        indices_to_add=["NDVI", "NDWI", "GNDVI"],
    )

    val_ds = CBERS4HydroGuidedDataset(
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
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def _normalize_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _resolve_fusion_weights(hydro_weight: float, ndwi_weight: float):
    hydro_weight = max(0.0, float(hydro_weight))
    ndwi_weight = max(0.0, float(ndwi_weight))
    guide_sum = hydro_weight + ndwi_weight

    if guide_sum >= 1.0:
        # Garante contribuicao da probabilidade da rede.
        scale = 0.95 / guide_sum
        hydro_weight *= scale
        ndwi_weight *= scale

    prob_weight = 1.0 - hydro_weight - ndwi_weight
    return prob_weight, hydro_weight, ndwi_weight


def _guided_binarize_hydro_ndwi(
    outputs: torch.Tensor,
    hydro_maps: torch.Tensor,
    images_wo_hydro: torch.Tensor,
    hydro_weight: float = HYDRO_WEIGHT,
    ndwi_weight: float = NDWI_WEIGHT,
    threshold: float = 0.5,
):
    """
    Funde probabilidade da rede com priors de hidrografia e NDWI.
    images_wo_hydro possui canais [B, G, R, NIR, NDVI, NDWI, GNDVI],
    entao NDWI esta no indice 5.
    """
    out_np = outputs.detach().cpu().numpy()
    hydro_np = hydro_maps.detach().cpu().numpy()
    img_np = images_wo_hydro.detach().cpu().numpy()

    preds_np = np.zeros_like(out_np, dtype=np.uint8)
    prob_w, hydro_w, ndwi_w = _resolve_fusion_weights(hydro_weight, ndwi_weight)

    for i in range(out_np.shape[0]):
        prob = np.clip(out_np[i, 0], 0.0, 1.0)
        hydro = _normalize_2d(hydro_np[i, 0])
        ndwi = _normalize_2d(img_np[i, 5])

        fused_score = prob_w * prob + hydro_w * hydro + ndwi_w * ndwi
        preds_np[i, 0] = (fused_score >= threshold).astype(np.uint8)

    preds = torch.from_numpy(preds_np).to(outputs.device)
    return preds


def compute_wiou(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    smooth: float = 1e-7,
    weight_borders: bool = True,
):
    """
    Calcula Weighted IoU (WIoU) onde bordas têm peso maior.
    predictions: (B, H, W) ou (B, 1, H, W) com valores 0/1
    ground_truth: (B, H, W) ou (B, 1, H, W) com valores 0/1
    weight_borders: se True, usa distance transform para dar peso às bordas
    """
    pred_flat = predictions.astype(np.float32).reshape(-1)
    gt_flat = ground_truth.astype(np.float32).reshape(-1)

    if weight_borders and ground_truth.sum() > 0:
        # Criar distance map para dar peso às bordas
        distance_map = np.zeros_like(ground_truth, dtype=np.float32)
        for i in range(ground_truth.shape[0]):
            if ground_truth[i].sum() > 0:
                dist = ndimage.distance_transform_edt(ground_truth[i] > 0.5)
                distance_map[i] = dist / (dist.max() + 1e-7)
        
        weights = 1.0 + distance_map.reshape(-1)
    else:
        weights = np.ones_like(pred_flat)

    intersection = np.sum(pred_flat * gt_flat * weights)
    union = np.sum((pred_flat + gt_flat - pred_flat * gt_flat) * weights)

    wiou = (intersection + smooth) / (union + smooth)
    return wiou


def save_predictions(model, loader, device, num_examples=5, results_dir="./results"):
    print("Salvando exemplos de previsoes com fusao guiada por hidrografia + NDWI...")
    model.eval()

    if len(loader) == 0:
        print("Loader de validacao vazio. Pulando.")
        return

    images, masks, hydro = next(iter(loader))
    images, masks, hydro = images.to(device), masks.to(device), hydro.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = _guided_binarize_hydro_ndwi(outputs, hydro, images)

    images_cpu = images.cpu()
    masks_cpu = masks.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    hydro_cpu = hydro.cpu().numpy()

    for i in range(min(num_examples, len(images_cpu))):
        img_rgb = images_cpu[i][[2, 1, 0], :, :].permute(1, 2, 0).numpy()
        img_rgb = _normalize_2d(img_rgb)
        ndwi_map = _normalize_2d(images_cpu[i][5].numpy())

        fig, axes = plt.subplots(1, 5, figsize=(24, 5))

        axes[0].imshow(np.clip(img_rgb, 0, 1))
        axes[0].set_title("Imagem")
        axes[0].axis("off")

        axes[1].imshow(hydro_cpu[i].squeeze(), cmap="Blues")
        axes[1].set_title("Hidrografia (guia)")
        axes[1].axis("off")

        axes[2].imshow(ndwi_map, cmap="viridis")
        axes[2].set_title("NDWI (guia)")
        axes[2].axis("off")

        axes[3].imshow(masks_cpu[i].squeeze(), cmap="gray")
        axes[3].set_title("Mascara Real")
        axes[3].axis("off")

        axes[4].imshow(preds_cpu[i].squeeze(), cmap="gray")
        axes[4].set_title("Predicao Fused")
        axes[4].axis("off")

        pred_path = os.path.join(results_dir, f"prediction_example_{i + 1}.png")
        plt.savefig(pred_path)
        plt.close(fig)


def execute_deeplab_train(model, device, train_loader, val_loader, loss_fn, num_epochs: int, results_dir="./results"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou_fused": [],
        "val_wiou_fused": [],
    }

    best_wiou = 0.0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")

        for images, masks, _hydro in pbar:
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_iou_fused = 0.0
        val_wiou_fused = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
            for images, masks, hydro in vbar:
                images, masks, hydro = images.to(device), masks.to(device), hydro.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                preds_fused = _guided_binarize_hydro_ndwi(outputs, hydro, images)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds_fused.float(), masks.long(), mode="binary", threshold=0.5
                )
                iou_fused = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                val_iou_fused += iou_fused.item()

                # Calcular WIoU
                wiou = compute_wiou(
                    preds_fused.cpu().numpy(),
                    masks.cpu().numpy(),
                    weight_borders=True,
                )
                val_wiou_fused += wiou

                vbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "iou": f"{iou_fused.item():.4f}",
                    "wiou": f"{wiou:.4f}",
                })

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou_fused = val_iou_fused / len(val_loader)
        avg_val_wiou_fused = val_wiou_fused / len(val_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_iou_fused"].append(avg_val_iou_fused)
        history["val_wiou_fused"].append(avg_val_wiou_fused)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} -> "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Fused IoU: {avg_val_iou_fused:.4f}, "
            f"Val WIoU: {avg_val_wiou_fused:.4f}"
        )

        if avg_val_wiou_fused > best_wiou:
            best_wiou = avg_val_wiou_fused
            best_epoch = epoch + 1
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Novo melhor modelo salvo em {model_path} com WIoU: {best_wiou:.4f}")

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
            plt.plot(history["val_iou_fused"], label="Validation Fused IoU", color="green")
            plt.title("Curva de Fused IoU")
            plt.xlabel("Epocas")
            plt.ylabel("Fused IoU")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            learning_curve_path = os.path.join(results_dir, "learning_curve.png")
            plt.savefig(learning_curve_path)
            plt.close()

    print(f"\nTreinamento concluido. Melhor WIoU: {best_wiou:.4f}")

    best_model_path = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        save_predictions(model, val_loader, device, num_examples=6, results_dir=results_dir)

    history_path = os.path.join(results_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return {
        "best_wiou": best_wiou,
        "best_iou": avg_val_iou_fused,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_wiou": avg_val_wiou_fused,
    }


def combined_loss(y_pred, y_true, fn_1="dice", fn_2="bce"):
    fns = {
        "dice": smp.losses.DiceLoss(mode="binary"),
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "focal": smp.losses.FocalLoss(mode="binary"),
        # "topk": smp.losses.TopKLoss(smp.losses.SoftBCEWithLogitsLoss(), k=0.2, mode='binary'),
    }

    return 0.5 * fns[fn_1](y_pred, y_true) + 0.5 * fns[fn_2](y_pred, y_true)


trains = {
    # "train_doce_test_itapemirim": {
    #     "train_dirs": ["../datasets/doce_cbers_mux"],
    #     "test_dirs": ["../datasets/itapemirim_cbers_mux"],
    # },
    # "train_itapemirim_test_doce": {
    #     "train_dirs": ["../datasets/itapemirim_cbers_mux"],
    #     "test_dirs": ["../datasets/doce_cbers_mux"],
    # },
    # "itapemirim": {
    #     "train_dirs": ["../datasets/itapemirim_cbers_mux"],
    #     "test_dirs": None,
    # },
    # "doce": {
    #     "train_dirs": ["../datasets/doce_cbers_mux"],
    #     "test_dirs": None,
    # },
    # "itapemirim_doce": {
    #     "train_dirs": ["../datasets/itapemirim_cbers_mux", "../datasets/doce_cbers_mux"],
    #     "test_dirs": None,
    # },
    "itapemirim_doce": {
        "train_dirs": ["../datasets/itapemirim_256", "../datasets/doce_256"],
        "test_dirs": None,
    },
}

backbones = [
    "resnext101_32x8d",
    # "efficientnet-b5",
]

all_results_summary = []

for train_num in range(1):
    for train_name, config in trains.items():
        for backbone in backbones:
            print(
                f"\n{'=' * 80}\nExperimento: {train_name} | Backbone: {backbone} "
                f"[DeepLabV3 Hydro+NDWI Guided No-Crop]\n{'=' * 80}"
            )

            train_red, train_nir, train_blue, train_green, train_masks, train_hydro = [], [], [], [], [], []
            for data_dir in config["train_dirs"]:
                r, n, b, g, m, h = load_cbers4_dataset_with_hydro(data_dir)
                train_red.extend(r)
                train_nir.extend(n)
                train_blue.extend(b)
                train_green.extend(g)
                train_masks.extend(m)
                train_hydro.extend(h)

            if config["test_dirs"] is None:
                (
                    train_red_paths,
                    val_red_paths,
                    train_nir_paths,
                    val_nir_paths,
                    train_blue_paths,
                    val_blue_paths,
                    train_green_paths,
                    val_green_paths,
                    train_mask_paths,
                    val_mask_paths,
                    train_hydro_paths,
                    val_hydro_paths,
                ) = train_test_split(
                    train_red,
                    train_nir,
                    train_blue,
                    train_green,
                    train_masks,
                    train_hydro,
                    test_size=0.2,
                    random_state=42,
                )
            else:
                train_red_paths = train_red
                train_nir_paths = train_nir
                train_blue_paths = train_blue
                train_green_paths = train_green
                train_mask_paths = train_masks
                train_hydro_paths = train_hydro

                val_red_paths, val_nir_paths, val_blue_paths, val_green_paths, val_mask_paths, val_hydro_paths = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for data_dir in config["test_dirs"]:
                    r, n, b, g, m, h = load_cbers4_dataset_with_hydro(data_dir)
                    val_red_paths.extend(r)
                    val_nir_paths.extend(n)
                    val_blue_paths.extend(b)
                    val_green_paths.extend(g)
                    val_mask_paths.extend(m)
                    val_hydro_paths.extend(h)

            train_paths_dict = {
                "red": train_red_paths,
                "nir": train_nir_paths,
                "blue": train_blue_paths,
                "green": train_green_paths,
                "masks": train_mask_paths,
                "hydro": train_hydro_paths,
            }
            val_paths_dict = {
                "red": val_red_paths,
                "nir": val_nir_paths,
                "blue": val_blue_paths,
                "green": val_green_paths,
                "masks": val_mask_paths,
                "hydro": val_hydro_paths,
            }

            train_loader, val_loader = create_dataloaders(
                train_paths=train_paths_dict, val_paths=val_paths_dict, img_size=256, batch_size=16
            )

            # Entram 7 canais: [B, G, R, NIR, NDVI, NDWI, GNDVI]
            model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=7,
                classes=1,
                activation="sigmoid",
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            results_dir = (
                f"./experiments/stats_dl/{train_num}/"
                f"results_{train_name}_{backbone}_deeplabv3_hydro_ndwi_guided_no_crop"
            )

            final_metrics = execute_deeplab_train(
                model=model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=combined_loss,
                num_epochs=NUM_EPOCHS,
                results_dir=results_dir,
            )

            summary_line = (
                f"Experimento: {train_name:<30} | Backbone: {backbone:<18}\n"
                f"  -> Melhor WIoU de Validacao: {final_metrics['best_wiou']:.4f} "
                f"(na epoca {final_metrics['best_epoch']})\n"
                f"  -> IoU Final (Validacao): {final_metrics['best_iou']:.4f}\n"
                f"  -> Perda Final (Treino/Val): {final_metrics['final_train_loss']:.4f} / "
                f"{final_metrics['final_val_loss']:.4f}\n"
            )
            all_results_summary.append(summary_line)

            print(summary_line)

summary_filepath = "final_results_summary_hydro_ndwi_guided_no_crop_2.txt"
print(f"\nSalvando resumo final em: {summary_filepath}")

with open(summary_filepath, "w", encoding="utf-8") as f:
    f.write("--- RESUMO FINAL DOS EXPERIMENTOS (DEEPLABV3 HYDRO+NDWI GUIDED NO-CROP) ---\n")
    f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 40 + "\n\n")

    for summary in all_results_summary:
        f.write(summary)
        f.write("-" * 60 + "\n")

print("Resumo salvo com sucesso.")
