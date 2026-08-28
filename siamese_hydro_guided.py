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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import CBERS4MUXDatasetWHidrography

LEARNING_RATE = 1e-4
NUM_EPOCHS = 150
HYDRO_WEIGHT = 0.4
TRIPLET_WEIGHT = 0.1
TRIPLET_MARGIN = 0.2
TRIPLET_SAMPLES = 2048
EMBEDDING_DIM = 32


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
    """Carrega caminhos de bandas, máscara e hidrografia de um único diretório."""
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
    Dataset que retorna RGB, NIR+índices e hidrografia separados
    para a rede siamesa.
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
        # Cada amostra 256x256 vira 4 crops 128x128.
        return len(self.base) * 4

    def __getitem__(self, idx):
        base_idx = idx // 4
        crop_idx = idx % 4

        full_image, mask = self.base[base_idx]

        # Ordem dos canais vindos da base: [B, G, R, NIR, NDVI, NDWI, GNDVI, HYDRO]
        # Extrair componentes
        rgb = full_image[:3, :, :]  # [B, G, R]
        nir_indices = full_image[3:7, :, :]  # [NIR, NDVI, NDWI, GNDVI]
        hydro = full_image[7:8, :, :]  # [HYDRO]

        # Divide 256x256 em 4 quadrantes 128x128: TL, TR, BL, BR.
        h_mid = rgb.shape[1] // 2
        w_mid = rgb.shape[2] // 2

        crops = [
            (slice(0, h_mid), slice(0, w_mid)),  # top-left
            (slice(0, h_mid), slice(w_mid, None)),  # top-right
            (slice(h_mid, None), slice(0, w_mid)),  # bottom-left
            (slice(h_mid, None), slice(w_mid, None)),  # bottom-right
        ]

        h_slice, w_slice = crops[crop_idx]

        rgb = rgb[:, h_slice, w_slice]
        nir_indices = nir_indices[:, h_slice, w_slice]
        hydro = hydro[:, h_slice, w_slice]
        mask = mask[:, h_slice, w_slice]

        return rgb, nir_indices, hydro, mask


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


class SiameseHydroGuidedNet(nn.Module):
    """
    Rede Siamesa com 3 branches:
    - Branch RGB: entrada com 3 canais (R, G, B)
    - Branch NIR+Índices: entrada com 4 canais (NIR, NDVI, NDWI, GNDVI)
    - Branch Hidrografia: entrada com 1 canal
    
    Cada branch usa um encoder DeepLabV3Plus. Os features são fusionados
    e passam por um decoder compartilhado.
    """

    def __init__(self, encoder_name: str = "efficientnet-b7"):
        super().__init__()

        # Branch RGB (3 canais)
        self.branch_rgb = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=256,  # número de features extraídas
            activation=None,
        )

        # Branch NIR+Índices (4 canais)
        # Carregamos pesos imagenet no encoder RGB, depois adaptamos para 4 canais
        self.branch_nir = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,  # Inicializamos sem pesos pré-treinados
            in_channels=4,
            classes=256,
            activation=None,
        )
        # Copiar pesos do encoder RGB para os 3 primeiros canais do NIR
        self._init_branch_nir_from_rgb()

        # Branch Hidrografia (1 canal)
        self.branch_hydro = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=1,
            classes=256,
            activation=None,
        )
        # Copiar pesos do encoder RGB para o primeiro canal da hidrografia
        self._init_branch_hydro_from_rgb()

        # Camada de fusão: concatena 256*3 = 768 features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Head de embedding para loss de métrica por pixel.
        self.embedding_head = nn.Conv2d(256, EMBEDDING_DIM, kernel_size=1)

        # Decoder simples para segmentação final
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _init_branch_nir_from_rgb(self):
        """Copia pesos do encoder RGB para os 3 primeiros canais do branch NIR."""
        # Copiar o estado do branch_rgb para branch_nir
        state_dict_rgb = self.branch_rgb.state_dict()
        state_dict_nir = self.branch_nir.state_dict()

        for key in state_dict_nir.keys():
            if key in state_dict_rgb:
                rgb_param = state_dict_rgb[key]
                nir_param = state_dict_nir[key]

                if rgb_param.shape == nir_param.shape:
                    state_dict_nir[key] = rgb_param
                    continue

                # Adapta apenas a primeira convolução de entrada: 3 canais -> 4 canais.
                if (
                    len(rgb_param.shape) == 4
                    and len(nir_param.shape) == 4
                    and rgb_param.shape[0] == nir_param.shape[0]
                    and rgb_param.shape[2:] == nir_param.shape[2:]
                    and rgb_param.shape[1] == 3
                    and nir_param.shape[1] == 4
                ):
                    adapted = nir_param.clone()
                    adapted[:, :3, :, :] = rgb_param
                    adapted[:, 3:4, :, :] = rgb_param.mean(dim=1, keepdim=True)
                    state_dict_nir[key] = adapted

        self.branch_nir.load_state_dict(state_dict_nir)

    def _init_branch_hydro_from_rgb(self):
        """Copia pesos do encoder RGB para o branch Hidrografia."""
        state_dict_rgb = self.branch_rgb.state_dict()
        state_dict_hydro = self.branch_hydro.state_dict()

        for key in state_dict_hydro.keys():
            if key in state_dict_rgb:
                rgb_param = state_dict_rgb[key]
                hydro_param = state_dict_hydro[key]

                if rgb_param.shape == hydro_param.shape:
                    state_dict_hydro[key] = rgb_param
                    continue

                # Adapta apenas a primeira convolução de entrada: 3 canais -> 1 canal.
                if (
                    len(rgb_param.shape) == 4
                    and len(hydro_param.shape) == 4
                    and rgb_param.shape[0] == hydro_param.shape[0]
                    and rgb_param.shape[2:] == hydro_param.shape[2:]
                    and rgb_param.shape[1] == 3
                    and hydro_param.shape[1] == 1
                ):
                    adapted = hydro_param.clone()
                    adapted[:, 0:1, :, :] = rgb_param.mean(dim=1, keepdim=True)
                    state_dict_hydro[key] = adapted

        self.branch_hydro.load_state_dict(state_dict_hydro)

    def forward(self, rgb, nir_indices, hydro):
        """
        Args:
            rgb: Tensor (B, 3, H, W) - canais RGB
            nir_indices: Tensor (B, 4, H, W) - canais NIR + índices
            hydro: Tensor (B, 1, H, W) - canal hidrografia
        
        Returns:
            output: Tensor (B, 1, H, W) - mapa de segmentação
            embeddings: Tensor (B, EMBEDDING_DIM, H, W) - embeddings por pixel
        """
        # Extrair features de cada branch
        feat_rgb = self.branch_rgb(rgb)  # (B, 256, H, W)
        feat_nir = self.branch_nir(nir_indices)  # (B, 256, H, W)
        feat_hydro = self.branch_hydro(hydro)  # (B, 256, H, W)

        # Concatenar features
        fused_feat = torch.cat([feat_rgb, feat_nir, feat_hydro], dim=1)  # (B, 768, H, W)

        # Fusão
        fused_feat = self.fusion_conv(fused_feat)  # (B, 256, H, W)

        # Embeddings normalizados para Triplet Loss.
        embeddings = self.embedding_head(fused_feat)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Decoder
        output = self.decoder(fused_feat)  # (B, 1, H, W)

        return output, embeddings


def _normalize_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _guided_binarize(
    outputs: torch.Tensor,
    hydro_maps: torch.Tensor,
    hydro_weight: float = HYDRO_WEIGHT,
    threshold: float = 0.5,
):
    """
    Funde probabilidade da rede com prior de hidrografia e aplica limiar fixo.
    Retorna predição binária.
    """
    out_np = outputs.detach().cpu().numpy()
    hydro_np = hydro_maps.detach().cpu().numpy()

    preds_np = np.zeros_like(out_np, dtype=np.uint8)

    for i in range(out_np.shape[0]):
        prob = np.clip(out_np[i, 0], 0.0, 1.0)
        hydro = _normalize_2d(hydro_np[i, 0])
        fused_score = (1.0 - hydro_weight) * prob + hydro_weight * hydro
        preds_np[i, 0] = (fused_score >= threshold).astype(np.uint8)

    preds = torch.from_numpy(preds_np).to(outputs.device)
    return preds


def _sample_pixel_triplets(
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    n_triplets: int = TRIPLET_SAMPLES,
):
    """Amostra triplas (ancora, positivo, negativo) no nível de pixel."""
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


def compute_triplet_loss(
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    margin: float = TRIPLET_MARGIN,
    n_triplets: int = TRIPLET_SAMPLES,
):
    """Calcula Triplet Loss por pixel; retorna 0 quando não há triplas válidas."""
    anchor, positive, negative = _sample_pixel_triplets(embeddings, masks, n_triplets=n_triplets)
    if anchor is None:
        return embeddings.new_tensor(0.0)

    triplet_fn = nn.TripletMarginLoss(margin=margin, p=2)
    return triplet_fn(anchor, positive, negative)


def save_predictions(model, loader, device, num_examples=5, results_dir="./results"):
    print("Salvando exemplos de previsoes...")
    model.eval()

    if len(loader) == 0:
        print("Loader de validacao vazio. Pulando.")
        return

    rgb, nir_indices, hydro, masks = next(iter(loader))
    rgb, nir_indices, hydro, masks = (
        rgb.to(device),
        nir_indices.to(device),
        hydro.to(device),
        masks.to(device),
    )

    with torch.no_grad():
        outputs, _embeddings = model(rgb, nir_indices, hydro)
        preds = torch.sigmoid(outputs)

    rgb_cpu = rgb.cpu()
    masks_cpu = masks.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    hydro_cpu = hydro.cpu().numpy()

    for i in range(min(num_examples, len(rgb_cpu))):
        img_rgb = rgb_cpu[i].permute(1, 2, 0).numpy()
        img_rgb = _normalize_2d(img_rgb)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(np.clip(img_rgb, 0, 1))
        axes[0].set_title("Imagem RGB")
        axes[0].axis("off")

        axes[1].imshow(hydro_cpu[i].squeeze(), cmap="Blues")
        axes[1].set_title("Hidrografia (guia)")
        axes[1].axis("off")

        axes[2].imshow(masks_cpu[i].squeeze(), cmap="gray")
        axes[2].set_title("Mascara Real")
        axes[2].axis("off")

        axes[3].imshow(preds_cpu[i].squeeze(), cmap="gray")
        axes[3].set_title("Predicao Siamesa")
        axes[3].axis("off")

        pred_path = os.path.join(results_dir, f"prediction_example_{i + 1}.png")
        plt.savefig(pred_path)
        plt.close(fig)


def execute_siamese_train(
    model, device, train_loader, val_loader, loss_fn, num_epochs: int, results_dir="./results"
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = {
        "train_loss": [],
        "train_seg_loss": [],
        "train_triplet_loss": [],
        "val_loss": [],
        "val_seg_loss": [],
        "val_triplet_loss": [],
        "val_iou": [],
    }

    best_iou = 0.0
    best_epoch = -1
    best_val_seg_loss = float("inf")
    best_val_triplet_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_triplet_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")

        for rgb, nir_indices, hydro, masks in pbar:
            rgb, nir_indices, hydro, masks = (
                rgb.to(device),
                nir_indices.to(device),
                hydro.to(device),
                masks.to(device),
            )

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, embeddings = model(rgb, nir_indices, hydro)
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
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "seg": f"{seg_loss.item():.4f}",
                    "trip": f"{triplet_loss.item():.4f}",
                }
            )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_triplet_loss = train_triplet_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_seg_loss"].append(avg_train_seg_loss)
        history["train_triplet_loss"].append(avg_train_triplet_loss)

        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_triplet_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
            for rgb, nir_indices, hydro, masks in vbar:
                rgb, nir_indices, hydro, masks = (
                    rgb.to(device),
                    nir_indices.to(device),
                    hydro.to(device),
                    masks.to(device),
                )

                outputs, embeddings = model(rgb, nir_indices, hydro)
                seg_loss = loss_fn(outputs, masks)
                triplet_loss = compute_triplet_loss(embeddings, masks)
                loss = seg_loss + TRIPLET_WEIGHT * triplet_loss
                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_triplet_loss += triplet_loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode="binary", threshold=0.5)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                val_iou += iou.item()

                vbar.set_postfix(
                    {
                        "val_loss": f"{loss.item():.4f}",
                        "seg": f"{seg_loss.item():.4f}",
                        "trip": f"{triplet_loss.item():.4f}",
                        "iou": f"{iou.item():.4f}",
                    }
                )

        avg_val_loss = val_loss / len(val_loader)
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_triplet_loss = val_triplet_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_seg_loss"].append(avg_val_seg_loss)
        history["val_triplet_loss"].append(avg_val_triplet_loss)
        history["val_iou"].append(avg_val_iou)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} -> "
            f"Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, Trip: {avg_train_triplet_loss:.4f}), "
            f"Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, Trip: {avg_val_triplet_loss:.4f}), "
            f"Val IoU: {avg_val_iou:.4f}"
        )

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_epoch = epoch + 1
            best_val_seg_loss = avg_val_seg_loss
            best_val_triplet_loss = avg_val_triplet_loss
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Novo melhor modelo salvo em {model_path} com IoU: {best_iou:.4f}")

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

            learning_curve_path = os.path.join(results_dir, "learning_curve.png")
            plt.savefig(learning_curve_path)
            plt.close()

    print(f"\nTreinamento concluido. Melhor IoU: {best_iou:.4f}")

    best_model_path = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        save_predictions(model, val_loader, device, num_examples=6, results_dir=results_dir)

    history_path = os.path.join(results_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return {
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_train_seg_loss": avg_train_seg_loss,
        "final_train_triplet_loss": avg_train_triplet_loss,
        "final_val_seg_loss": avg_val_seg_loss,
        "final_val_triplet_loss": avg_val_triplet_loss,
        "best_val_seg_loss": best_val_seg_loss,
        "best_val_triplet_loss": best_val_triplet_loss,
    }


def combined_loss(y_pred, y_true, fn_1="dice", fn_2="bce"):
    fns = {
        "dice": smp.losses.DiceLoss(mode="binary"),
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "focal": smp.losses.FocalLoss(mode="binary"),
    }

    return 0.5 * fns[fn_1](y_pred, y_true) + 0.5 * fns[fn_2](y_pred, y_true)


trains = {
    "itapemirim_doce": {
        "train_dirs": ["../datasets/itapemirim_256", "../datasets/doce_256"],
        "test_dirs": None,
    }
}

backbones = ["resnet152"]

all_results_summary = []

for train_num in range(1):
    for train_name, config in trains.items():
        for backbone in backbones:
            print(f"\n{'=' * 80}\nExperimento: {train_name} | Backbone: {backbone} [Siamese Hydro-Guided]\n{'=' * 80}")

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
                train_paths=train_paths_dict, val_paths=val_paths_dict, img_size=256, batch_size=8
            )

            # Criar modelo siamesa
            model = SiameseHydroGuidedNet(encoder_name=backbone)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            results_dir = f"./experiments/stats_siamese/{train_num}/results_{train_name}_{backbone}_siamese_hydro_guided"

            final_metrics = execute_siamese_train(
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
                f"  -> Melhor IoU de Validacao: {final_metrics['best_iou']:.4f} "
                f"(na epoca {final_metrics['best_epoch']})\n"
                f"  -> Perda Final (Treino/Val): {final_metrics['final_train_loss']:.4f} / "
                f"{final_metrics['final_val_loss']:.4f}\n"
                f"  -> Perda Final Seg (Treino/Val): {final_metrics['final_train_seg_loss']:.4f} / "
                f"{final_metrics['final_val_seg_loss']:.4f}\n"
                f"  -> Perda Final Triplet (Treino/Val): {final_metrics['final_train_triplet_loss']:.4f} / "
                f"{final_metrics['final_val_triplet_loss']:.4f}\n"
                f"  -> Val Seg/Triplet na melhor IoU: {final_metrics['best_val_seg_loss']:.4f} / "
                f"{final_metrics['best_val_triplet_loss']:.4f}\n"
            )
            all_results_summary.append(summary_line)

            print(summary_line)

summary_filepath = "final_results_summary_siamese.txt"
print(f"\nSalvando resumo final em: {summary_filepath}")

with open(summary_filepath, "w", encoding="utf-8") as f:
    f.write("--- RESUMO FINAL DOS EXPERIMENTOS (SIAMESE HYDRO-GUIDED) ---\n")
    f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 40 + "\n\n")

    for summary in all_results_summary:
        f.write(summary)
        f.write("-" * 60 + "\n")

print("Resumo salvo com sucesso.")
