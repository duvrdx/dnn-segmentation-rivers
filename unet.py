import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from datetime import datetime 


from datasets import CBERS4MUXDataset

LEARNING_RATE = 1e-4

def load_cbers4_dataset(data_dir: str):
    """Carrega os caminhos dos arquivos de um único diretório de dataset."""
    RED_DIR = os.path.join(data_dir, 'red')
    GREEN_DIR = os.path.join(data_dir, 'green')
    BLUE_DIR = os.path.join(data_dir, 'blue')
    NIR_DIR = os.path.join(data_dir, 'nir')
    MASK_DIR = os.path.join(data_dir, 'masks')
    
    ids = sorted([f for f in os.listdir(RED_DIR) if f.endswith('.tiff') or f.endswith('.tif')])
    
    red_paths = [os.path.join(RED_DIR, f) for f in ids]
    nir_paths = [os.path.join(NIR_DIR, f) for f in ids]
    blue_paths = [os.path.join(BLUE_DIR, f) for f in ids]
    green_paths = [os.path.join(GREEN_DIR, f) for f in ids]
    mask_paths = [os.path.join(MASK_DIR, f) for f in ids]
    
    return red_paths, nir_paths, blue_paths, green_paths, mask_paths

def create_dataloaders(
    train_paths: dict, 
    val_paths: dict, 
    img_size: int = 128, 
    batch_size: int = 8
):
    """
    Cria DataLoaders de treino e validação a partir de dicionários de caminhos de arquivos.
    """
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
    
    train_loader_dataset = CBERS4MUXDataset(
        red_image_paths=train_paths['red'],
        nir_image_paths=train_paths['nir'],
        blue_image_paths=train_paths['blue'],
        green_image_paths=train_paths['green'],
        mask_paths=train_paths['masks'],
        transform=train_transform,
        indices_to_add=['NDVI', 'NDWI', 'GNDVI']
    )
    
    val_loader_dataset = CBERS4MUXDataset(
        red_image_paths=val_paths['red'],
        nir_image_paths=val_paths['nir'],
        blue_image_paths=val_paths['blue'],
        green_image_paths=val_paths['green'],
        mask_paths=val_paths['masks'],
        transform=val_transform,
        indices_to_add=['NDVI', 'NDWI', 'GNDVI']
    )

    train_loader = DataLoader(train_loader_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_loader_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def save_predictions(model, loader, device, num_examples=5, results_dir="./results"):
    """
    Salva exemplos de previsão do primeiro batch do loader.
    """
    print("Salvando exemplos de previsões...")
    model.eval()
    
    if len(loader) == 0:
        print("Loader de validação está vazio. Pulando a geração de exemplos.")
        return
        
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = (outputs > 0.5).int()

    images_cpu = images.cpu()
    masks_cpu = masks.cpu().numpy()
    preds_cpu = preds.cpu().numpy()

    for i in range(min(num_examples, len(images_cpu))):
        img_rgb_normalized = images_cpu[i][[2, 1, 0], :, :]
        img_rgb_normalized = img_rgb_normalized.permute(1, 2, 0).numpy()

        min_val = img_rgb_normalized.min()
        max_val = img_rgb_normalized.max()
        if max_val > min_val:
            img_to_show = (img_rgb_normalized - min_val) / (max_val - min_val)
        else:
            img_to_show = np.zeros_like(img_rgb_normalized)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(np.clip(img_to_show, 0, 1))
        ax1.set_title("Imagem Original")
        ax1.axis('off')

        ax2.imshow(masks_cpu[i].squeeze(), cmap='gray')
        ax2.set_title("Máscara Real (Ground Truth)")
        ax2.axis('off')

        ax3.imshow(preds_cpu[i].squeeze(), cmap='gray')
        ax3.set_title("Máscara Prevista")
        ax3.axis('off')

        pred_path = os.path.join(results_dir, f'prediction_example_{i+1}.png')
        plt.savefig(pred_path)
        plt.close(fig)
    
    print(f"{min(num_examples, len(images_cpu))} exemplos salvos no diretório '{results_dir}'.")
    
def execute_unet_train(model: smp.Unet, device: torch.device, train_loader: DataLoader,
                       val_loader: DataLoader, loss_fn, num_epochs: int, results_dir="./results"):
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) 
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    print(f"Iniciando treinamento no dispositivo: {device}")

    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0.0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            
            for images, masks in vbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks.long(), mode='binary', threshold=0.5)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
                val_iou += iou.item()
                vbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_iou': f'{iou.item():.4f}'})
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_epoch = epoch + 1
            os.makedirs(results_dir, exist_ok=True) 
            model_path = os.path.join(results_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Novo melhor modelo salvo em {model_path} com IoU: {best_iou:.4f}")
            
        scheduler.step(avg_val_loss)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title("Curva de Perda (Loss)")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history['val_iou'], label='Validation IoU', color='green')
        plt.title("Curva de IoU de Validação")
        plt.xlabel("Épocas")
        plt.ylabel("IoU Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        learning_curve_path = os.path.join(results_dir, 'learning_curve.png')
        plt.savefig(learning_curve_path)
        plt.close()

    print(f"\nTreinamento concluído. Melhor IoU alcançada: {best_iou:.4f}")
    
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"\nCarregando o melhor modelo de '{best_model_path}' para gerar exemplos.")
        model.load_state_dict(torch.load(best_model_path))
        save_predictions(model, val_loader, device, 6, results_dir)
    else:
        print("Nenhum modelo foi salvo. Pulando a geração de exemplos.")
        
    piclk_path = os.path.join(results_dir, 'training_history.pkl')
    with open(piclk_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Histórico de treinamento salvo em {piclk_path}")

    return {
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }

loss_fn_dice = smp.losses.DiceLoss(mode='binary')
loss_fn_bce = smp.losses.SoftBCEWithLogitsLoss()

def combined_loss(y_pred, y_true):
    return 0.5 * loss_fn_dice(y_pred, y_true) + 0.5 * loss_fn_bce(y_pred, y_true)

# --- Configuração dos Treinamentos ---
trains = {
    'train_doce_test_itapemirim': {
        'train_dirs': ['./datasets/doce_cbers_mux'],
        'test_dirs': ['./datasets/itapemirim_cbers_mux']
    },
    'train_itapemirim_test_doce': {
        'train_dirs': ['./datasets/itapemirim_cbers_mux'],
        'test_dirs': ['./datasets/doce_cbers_mux']
    },
    'itapemirim': {
        'train_dirs': ['./datasets/itapemirim_cbers_mux'],
        'test_dirs': None
    },
    'doce': {
        'train_dirs': ['./datasets/doce_cbers_mux'],
        'test_dirs': None
    },
    'itapemirim_doce': {
        'train_dirs': ['./datasets/itapemirim_cbers_mux', './datasets/doce_cbers_mux'],
        'test_dirs': None
    }    
}

backbones = [
    "resnet152",
    "efficientnet-b5"
]

all_results_summary = []

for train_num in range(10):
    for train_name, config in trains.items():
        for backbone in backbones:
            print(f"\n{'='*80}\n--- Iniciando Experimento: Treino='{train_name}', Backbone='{backbone}' ---\n{'='*80}")
            
            train_red, train_nir, train_blue, train_green, train_masks = [], [], [], [], []
            for data_dir in config['train_dirs']:
                r, n, b, g, m = load_cbers4_dataset(data_dir)
                train_red.extend(r); train_nir.extend(n); train_blue.extend(b); train_green.extend(g); train_masks.extend(m)

            if config['test_dirs'] is None:
                print(f"Dividindo o dataset '{train_name}' em treino e validação (80/20).")
                (train_red_paths, val_red_paths, train_nir_paths, val_nir_paths, train_blue_paths, val_blue_paths,
                train_green_paths, val_green_paths, train_mask_paths, val_mask_paths) = train_test_split(
                    train_red, train_nir, train_blue, train_green, train_masks, test_size=0.2, random_state=42)
            else:
                test_dataset_name = os.path.basename(config['test_dirs'][0])
                print(f"Usando '{train_name}' para treino e '{test_dataset_name}' para validação.")
                train_red_paths, train_nir_paths, train_blue_paths, train_green_paths, train_mask_paths = train_red, train_nir, train_blue, train_green, train_masks
                
                val_red_paths, val_nir_paths, val_blue_paths, val_green_paths, val_mask_paths = [], [], [], [], []
                for data_dir in config['test_dirs']:
                    r, n, b, g, m = load_cbers4_dataset(data_dir)
                    val_red_paths.extend(r); val_nir_paths.extend(n); val_blue_paths.extend(b); val_green_paths.extend(g); val_mask_paths.extend(m)

            train_paths_dict = {'red': train_red_paths, 'nir': train_nir_paths, 'blue': train_blue_paths, 'green': train_green_paths, 'masks': train_mask_paths}
            val_paths_dict = {'red': val_red_paths, 'nir': val_nir_paths, 'blue': val_blue_paths, 'green': val_green_paths, 'masks': val_mask_paths}
            
            train_loader, val_loader = create_dataloaders(
                train_paths=train_paths_dict, val_paths=val_paths_dict, img_size=128, batch_size=8)
            
            model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet", in_channels=7, classes=1, activation='sigmoid')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            results_dir = f'./experiments/stats___/{train_num}/results_{train_name}_{backbone}'
            
            final_metrics = execute_unet_train(
                model=model, device=device, train_loader=train_loader, val_loader=val_loader, 
                loss_fn=combined_loss, num_epochs=50, results_dir=results_dir
            )
            
            summary_line = (
                f"Experimento: {train_name:<30} | Backbone: {backbone:<18}\n"
                f"  -> Melhor IoU de Validação: {final_metrics['best_iou']:.4f} (na época {final_metrics['best_epoch']})\n"
                f"  -> Perda Final (Treino/Val): {final_metrics['final_train_loss']:.4f} / {final_metrics['final_val_loss']:.4f}\n"
            )
            all_results_summary.append(summary_line)
            
            print(f"--- Resumo do Experimento ---\n{summary_line}")
            print(f"--- Experimento '{train_name}' com backbone '{backbone}' concluído. ---")

summary_filepath = 'final_results_summary___.txt'
print(f"\n{'='*80}\nSalvando resumo final de todos os experimentos em: {summary_filepath}\n{'='*80}")

with open(summary_filepath, 'w', encoding='utf-8') as f:
    f.write("--- RESUMO FINAL DOS EXPERIMENTOS ---\n")
    f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*40 + "\n\n")
    
    for summary in all_results_summary:
        f.write(summary)
        f.write("-" * 60 + "\n")
        
print("Resumo salvo com sucesso.")