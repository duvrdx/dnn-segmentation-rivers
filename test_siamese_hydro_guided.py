import os
import pickle
from datetime import datetime

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from siamese_hydro_guided import (
    SiameseHydroGuidedNet,
    CBERS4HydroGuidedDataset,
    combined_loss,
    compute_triplet_loss,
    _guided_binarize,
    save_predictions,
    _find_hydro_dir,
    NUM_EPOCHS,
    LEARNING_RATE,
    HYDRO_WEIGHT,
    TRIPLET_WEIGHT,
)


def load_cbers4_dataset_with_hydro(data_dir: str):
    red_dir = os.path.join(data_dir, "red")
    green_dir = os.path.join(data_dir, "green")
    blue_dir = os.path.join(data_dir, "blue")
    nir_dir = os.path.join(data_dir, "nir")
    mask_dir = os.path.join(data_dir, "masks_ibge")
    hydro_dir = _find_hydro_dir(data_dir)

    ids = sorted([f for f in os.listdir(red_dir) if f.endswith(".tiff") or f.endswith(".tif")])

    return (
        [os.path.join(red_dir, f) for f in ids],
        [os.path.join(nir_dir, f) for f in ids],
        [os.path.join(blue_dir, f) for f in ids],
        [os.path.join(green_dir, f) for f in ids],
        [os.path.join(mask_dir, f) for f in ids],
        [os.path.join(hydro_dir, f) for f in ids],
    )


def create_dataloaders(train_paths, val_paths, img_size=256, batch_size=8):
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
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader


def train_simple(
    model, device, train_loader, val_loader, loss_fn,
    num_epochs, results_dir, backbone,
):
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
        train_loss = 0.0
        train_seg_loss = 0.0
        train_triplet_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{num_epochs} [Train]")

        for rgb, nir_indices, hydro, masks in pbar:
            rgb = rgb.to(device)
            nir_indices = nir_indices.to(device)
            hydro = hydro.to(device)
            masks = masks.to(device)

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
            for rgb, nir_indices, hydro, masks in vbar:
                rgb = rgb.to(device)
                nir_indices = nir_indices.to(device)
                hydro = hydro.to(device)
                masks = masks.to(device)

                outputs, embeddings = model(rgb, nir_indices, hydro)
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

    parser = argparse.ArgumentParser(description="Teste rapido Siamese Hydro-Guided")
    parser.add_argument("--data_dir", type=str, default="../datasets/doce_256",
                        help="Diretorio do dataset")
    parser.add_argument("--backbone", type=str, default="resnet152",
                        choices=["timm-efficientnet-b7", "mit_b5", "resnet152", "efficientnet-b5"])
    parser.add_argument("--epochs", type=int, default=3,
                        help="Numero de epocas")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proporcao para validacao")
    parser.add_argument("--results_dir", type=str, default="./test_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Epocas: {args.epochs}")
    print(f"Batch: {args.batch_size}")
    print(f"Val split: {args.val_split}")

    red, nir, blue, green, masks, hydro = load_cbers4_dataset_with_hydro(args.data_dir)
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
        "hydro": [hydro[i] for i in train_idx],
    }
    val_paths = {
        "red": [red[i] for i in val_idx],
        "nir": [nir[i] for i in val_idx],
        "blue": [blue[i] for i in val_idx],
        "green": [green[i] for i in val_idx],
        "masks": [masks[i] for i in val_idx],
        "hydro": [hydro[i] for i in val_idx],
    }

    print(f"Treino: {len(train_idx)} amostras (x4 crops = {len(train_idx) * 4} tiles) | "
          f"Val: {len(val_idx)} amostras (x4 crops = {len(val_idx) * 4} tiles)")

    train_loader, val_loader = create_dataloaders(
        train_paths, val_paths, img_size=128, batch_size=args.batch_size
    )

    model = SiameseHydroGuidedNet(encoder_name=args.backbone)

    run_name = f"{args.backbone}_siamese_hydro_guided_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
