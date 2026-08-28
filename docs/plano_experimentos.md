# Plano de Experimentos — Siamese Cross-Attention com K-Fold

## Sumário

1. [Metodologia K-Fold](#1-metodologia-k-fold)
2. [Arquitetura Siamese Cross-Attention](#2-arquitetura-siamese-cross-attention)
3. [Datasets e Configurações](#3-datasets-e-configurações)
4. [Backbones](#4-backbones)
5. [Hyperparâmetros](#5-hiperparâmetros)
6. [Métricas de Avaliação](#6-métricas-de-avaliação)
7. [Estrutura dos Experimentos](#7-estrutura-dos-experimentos)
8. [Comparações com Baselines Existentes](#8-comparações-com-baselines-existentes)
9. [Estrutura de Diretórios](#9-estrutura-de-diretórios)

---

## 1. Metodologia K-Fold

### Problema do Método Atual

```python
for train_num in range(10):              # 10 repetições
    train_test_split(test_size=0.2,       # mesmo split 80/20
                     random_state=42)     # mesma semente!
```

Este loop treina 10 modelos na **mesma divisão** de dados — não há variabilidade entre as execuções. A estimativa de performance é pontual, sem intervalo de confiança.

### Método Proposto: K-Fold Cross-Validation (K=5)

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    # Fold k: treina em 80%, valida em 20% (DIFERENTE a cada fold)
    train_model(data[train_idx], data[val_idx])
```

**Vantagens** (Kohavi, 1995 — IJCAI):
- Cada amostra é usada para validação exatamente uma vez
- Relatório com **média ± desvio padrão** (intervalo de confiança)
- Detecta overfitting a um split específico
- K=5 escolhido por: mesmo proporção 80/20 do método atual, custo computacional 5x (vs 10x), recomendado empiricamente

### Casos de Uso do K-Fold

| Configuração | K-Fold? | Motivo |
|---|---|---|
| `doce` | Sim | Dataset único, divisão aleatória |
| `itapemirim` | Sim | Dataset único, divisão aleatória |
| `itapemirim_doce` | Sim | Dataset combinado, manter proporção |
| `train_doce_test_itapemirim` | **Não** | Cenário de generalização cruzada (hold-out) |
| `train_itapemirim_test_doce` | **Não** | Cenário de generalização cruzada (hold-out) |

---

## 2. Arquitetura Siamese Cross-Attention

### Diagrama da Arquitetura

```
Entrada:            RGB (3 canais)      NIR+Índices (7 canais)    Hidrografia (1 canal)
                         │                      │                        │
                         ▼                      ▼                        ▼
Encoders SMP:    ┌──────────────┐    ┌──────────────┐           ┌──────────────┐
                 │  mit_b5 /    │    │  mit_b5 /    │           │  mit_b5 /    │
                 │ effnet-b7 /  │    │ effnet-b7 /  │           │ effnet-b7 /  │
                 │ resnet152    │    │ resnet152    │           │ resnet152    │
                 └──────┬───────┘    └──────┬───────┘           └──────┬───────┘
                        │                    │                         │
               Features multi-escala  Features multi-escala   Features multi-escala
                (níveis 2,3,4,5)       (níveis 2,3,4,5)        (níveis 2,3,4,5)
                        │                    │                         │
                        └──────────┬─────────┘                         │
                                   │                                   │
                                   ▼                                   │
                     ┌─────────────────────────┐                       │
                     │   Cross-Attention RGB   │                       │
                     │   ↔ NIR+Índices         │                       │
                     │   (por nível)           │                       │
                     └────────────┬────────────┘                       │
                                  │                                    │
                                  ▼                                    ▼
                     ┌─────────────────────────┐              ┌──────────────┐
                     │  Features Fundidas      │◄────FiLM────│  Hidrografia │
                     │  (4 níveis)             │    γ,β       │  (Conv1x1)   │
                     └────────────┬────────────┘              └──────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Decoder U-Net-like    │
                     │   (upsampling + skip)   │
                     │   Deep supervision ↓    │
                     │   - Loss aux 16x16      │
                     │   - Loss aux 32x32      │
                     │   - Loss aux 64x64      │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Saída (1, 256, 256)   │
                     │   + Embeddings (32 dim) │
                     │   para Triplet Loss     │
                     └─────────────────────────┘
```

### Componentes Detalhados

#### 2.1 EncoderBranch
- SMP encoder puro (sem decoder) — extrai features dos níveis 2, 3, 4, 5
- Pesos ImageNet para RGB, transferidos para os outros branches via adaptação de canais

#### 2.2 CrossModalFusionBlock (por nível de resolução)
- **Entrada:** features RGB nível i, features NIR nível i
- **Cross-attention bidirecional:**
  - Q = RGB, K = V = NIR (destaque de regiões onde NIR confirma/refuta RGB)
  - Q = NIR, K = V = RGB (complementar)
- **Saída:** feature fundida do mesmo tamanho

#### 2.3 FiLM Conditioning
- **Entrada:** feature de hidrografia (via encoder leve)
- **Processamento:** Conv1x1 → γ, σ por canal
- **Modulação:** `f_out = γ * f_fundida + σ`
- **Efeito:** permite que a hidrografia amplifique ou suprima canais específicos

#### 2.4 Decoder U-Net + Deep Supervision
- Upsampling progressivo com skip connections das features fundidas
- Perdas auxiliares nos níveis 16×16, 32×32, 64×64
- Perda principal no nível 256×256

#### 2.5 Hard Triplet Mining
- Batch-hard mining: para cada pixel âncora de água, hardest positive + hardest negative
- Embeddings normalizados (32 dim) para cálculo de distância coseno
- Triplet margin = 0.5, weight = 0.5

---

## 3. Datasets e Configurações

### Estrutura dos Dados

```
datasets/
├── doce_256/
│   ├── red/         *.tif/.tiff
│   ├── green/       *.tif/.tiff
│   ├── blue/        *.tif/.tiff
│   ├── nir/         *.tif/.tiff
│   ├── masks_ibge/  *.tif/.tiff
│   └── hidrography/ *.tif/.tiff
└── itapemirim_256/
    ├── red/
    ├── green/
    ├── blue/
    ├── nir/
    ├── masks_ibge/
    └── hidrography/
```

### Configurações de Dataset

| Nome | Diretórios | Amostras (aprox.) | Modo |
|---|---|---|---|
| `doce` | `[doce_256]` | ~100-200 | K-Fold |
| `itapemirim` | `[itapemirim_256]` | ~100-200 | K-Fold |
| `itapemirim_doce` | `[doce_256, itapemirim_256]` | ~200-400 | K-Fold |
| `train_doce_test_itapemirim` | train: `[doce_256]`, test: `[itapemirim_256]` | ~100-200 | Hold-out |
| `train_itapemirim_test_doce` | train: `[itapemirim_256]`, test: `[doce_256]` | ~100-200 | Hold-out |

### Dataset CBERS4MUXSiameseDataset

```python
class CBERS4MUXSiameseDataset(Dataset):
    """Retorna 4 tensores por amostra (256×256, sem crop)."""
    def __getitem__(self, idx):
        # rgb:        (3, 256, 256)  — B, G, R
        # spectral:   (7, 256, 256)  — B, G, R, NIR, NDVI, NDWI, GNDVI
        # hydro:      (1, 256, 256)  — mapa de hidrografia IBGE
        # mask:       (1, 256, 256)  — ground truth binário
        return rgb, spectral, hydro, mask
```

---

## 4. Backbones

| Backbone | Parâmetros do Encoder | Canais por Nível | Origem |
|---|---|---|---|
| `mit_b5` | ~83M | [64, 128, 320, 512] | SegFormer (Xie, 2021) |
| `timm-efficientnet-b7` | ~66M | [48, 80, 224, 640] | EfficientNet (Tan & Le, 2019) |
| `resnet152` | ~60M | [256, 512, 1024, 2048] | Deep Residual Learning (He, 2016) |

**Critério de escolha:**
- **mit_b5**: melhor custo-benefício entre transformers hierárquicos; já testado no projeto
- **efficientnet-b7**: backbone de melhor performance no baseline DeepLabV3+
- **resnet152**: backbone da siamesa atual; permite comparação direta

---

## 5. Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `LEARNING_RATE` | 1e-4 | Mesmo de todos os experimentos anteriores |
| `NUM_EPOCHS` | 150 | Suficiente para convergência (siamesa atual) |
| `OPTIMIZER` | AdamW | Mesmo dos experimentos anteriores |
| `SCHEDULER` | ReduceLROnPlateau (factor=0.1, patience=10) | Padrão do projeto |
| `BATCH_SIZE` | 8 (16 se memória permitir) | 3 branches com 256×256 |
| `LOSS_SEG` | 0.5 Dice + 0.5 BCE | Mesmo do projeto (combined_loss) |
| `TRIPLET_WEIGHT` | 0.5 | Aumentado de 0.1 (atual) para 0.5 |
| `TRIPLET_MARGIN` | 0.5 | Aumentado de 0.2 (atual) para 0.5 |
| `EMBEDDING_DIM` | 32 | Mesmo da siamesa atual |
| `NUM_FOLDS` (K) | 5 | Kohavi (1995) |
| `RANDOM_STATE` | 42 | Mesmo do projeto |
| `AMP` | True | Mixed precision (já implementado) |
| `WEIGHT_DECAY` | 1e-4 | Regularização |

---

## 6. Métricas de Avaliação

### 6.1 IoU (Intersection over Union)
```python
tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode="binary")
iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
```

### 6.2 WIoU (Weighted IoU)
```python
# Distância das bordas via distance_transform_edt (scipy)
# Pixels próximos à borda têm maior peso
boundary_weight = 1.0 + alpha * distance_transform(1 - mask)
wiou = sum(weighted_correct) / sum(weighted_total)
```
Implementado no script `deeplab_hydro_ndwi_guided_otsu.py`. Avalia qualidade das bordas da segmentação.

### 6.3 Otsu Threshold
Para cada imagem na validação, calcula o limiar de Otsu no mapa de probabilidade, em vez de usar threshold fixo 0.5. Demonstrou melhorar resultados nos experimentos com guidance.

### 6.4 Relatório K-Fold

Para cada experimento K-Fold, o relatório final será:

```
=== EXPERIMENTO: doce | Backbone: mit_b5 | K=5 ===
Fold 0: IoU = 0.8631, WIoU = 0.7542
Fold 1: IoU = 0.8714, WIoU = 0.7610
Fold 2: IoU = 0.8589, WIoU = 0.7489
Fold 3: IoU = 0.8657, WIoU = 0.7571
Fold 4: IoU = 0.8698, WIoU = 0.7603
-------------------------------------------------------------
Média IoU:  0.8658 ± 0.0051
Média WIoU: 0.7563 ± 0.0049
-------------------------------------------------------------
```

---

## 7. Estrutura dos Experimentos

### 7.1 Total de Execuções

| Dataset | Modo | Backbones | Execuções |
|---|---|---|---|
| `doce` | K-Fold (5) | 3 | 15 |
| `itapemirim` | K-Fold (5) | 3 | 15 |
| `itapemirim_doce` | K-Fold (5) | 3 | 15 |
| `train_doce_test_itapemirim` | Hold-out | 3 | 3 |
| `train_itapemirim_test_doce` | Hold-out | 3 | 3 |
| **Total** | | | **51** |

### 7.2 Algoritmo de Execução

```python
def main():
    for backbone in backbones:
        for config_name, config in experiments.items():
            if config["type"] == "kfold":
                # 1. Carregar paths
                all_paths = load_paths(config["train_dirs"])

                # 2. K-Fold splits
                splits = create_kfold_splits(all_paths, n_splits=5)

                # 3. Treinar cada fold
                for fold, (train_paths, val_paths) in enumerate(splits):
                    model = SiameseCrossAttentionNet(encoder_name=backbone)
                    metrics = train_fold(model, train_paths, val_paths, fold)

                # 4. Agregar resultados
                save_fold_summary(config_name, backbone, fold_metrics)

            else:  # hold-out
                train_paths = load_paths(config["train_dirs"])
                val_paths = load_paths(config["test_dirs"])
                model = SiameseCrossAttentionNet(encoder_name=backbone)
                metrics = train_holdout(model, train_paths, val_paths)
                save_holdout_summary(config_name, backbone, metrics)
```

---

## 8. Comparações com Baselines Existentes

### 8.1 Baselines do Projeto

| Modelo | Dataset | Melhor IoU | Arquivo de Resultados |
|---|---|---|---|
| DeepLabV3+ (sem guidance) | itapemirim_doce | 0.7966 | `final_results_summary.txt` |
| DeepLabV3+ (hydro-guided, crop) | itapemirim_doce | 0.7519 | `final_results_summary_hydro_guided.txt` |
| DeepLabV3+ (hydro-guided, no-crop) | doce | **0.8728** | `final_results_summary_hydro_guided_no_crop.txt` |
| DeepLabV3+ (hydro-guided, no-crop) | itapemirim_doce | **0.8520** | `final_results_summary_hydro_guided_no_crop.txt` |
| DeepLabV3+ (hydro+NDWI guided) | itapemirim_doce | 0.7875 | `final_results_summary_hydro_ndwi_guided_crop4.txt` |
| SegFormer (mit_b5) | itapemirim_doce | 0.8173 | `final_results_summary_segformer.txt` |
| Siamese Hydro-Guided (atual) | itapemirim_doce | 0.8271 | `final_results_summary_siamese.txt` |

### 8.2 Alvos da Proposta

| Dataset | Baseline (melhor) | Alvo Siamese Cross-Attention |
|---|---|---|
| doce | 0.8728 ± ? (DeepLabV3+ hydro-guided no-crop) | **>0.88** (superar com fusão aprendida) |
| itapemirim_doce | 0.8520 ± ? (DeepLabV3+ hydro-guided no-crop) | **>0.86** |
| train_doce_test_itapemirim | 0.6643 (DeepLabV3+ hydro-guided no-crop) | **>0.70** (generalização) |
| train_itapemirim_test_doce | 0.7136 (DeepLabV3+ hydro-guided no-crop) | **>0.75** (generalização) |

**Nota:** os valores dos baselines são pontuais (sem K-Fold). A comparação será mais robusta com os intervalos de confiança da média ± std.

---

## 9. Estrutura de Diretórios

```
dnn-segmentation-rivers/
├── docs/
│   ├── embasamento_teorico.md       ← Este documento descreve referências
│   └── plano_experimentos.md        ← Este documento
│
├── datasets.py                       ← + CBERS4MUXSiameseDataset
├── utils.py                          ← + create_kfold_splits(), compute_statistics()
├── siamese_cross_attention.py        ← NOVO: implementação completa
│
├── experiments/
│   └── stats_siamese_ca/             ← Resultados da nova siamesa
│       ├── doce/
│       │   ├── mit_b5/
│       │   │   ├── fold_0/   (best_model.pth, learning_curve.png, preds...)
│       │   │   ├── fold_1/
│       │   │   ├── fold_2/
│       │   │   ├── fold_3/
│       │   │   └── fold_4/
│       │   ├── timm-efficientnet-b7/
│       │   └── resnet152/
│       ├── itapemirim/
│       ├── itapemirim_doce/
│       ├── train_doce_test_itapemirim/   (hold-out, sem subpastas fold)
│       └── train_itapemirim_test_doce/   (hold-out, sem subpastas fold)
│
└── final_results_summary_siamese_ca.txt  ← Resultados agregados finais
```

---

## Referências

1. Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI.
2. Hermans, A., Beyer, L., & Leibe, B. (2017). *In Defense of the Triplet Loss for Person Re-Identification.* arXiv:1703.07737.
3. Perez, E., et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI. arXiv:1709.07871.
4. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
5. Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS. arXiv:2105.15203.
6. Chen, L.C., et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV.
7. McFeeters, S.K. (1996). *The use of NDWI in the delineation of open water features.* IJRS.
8. Lin, T.Y., et al. (2017). *Feature Pyramid Networks for Object Detection.* CVPR.
9. Lee, C.Y., et al. (2015). *Deeply-Supervised Nets.* AISTATS.
10. Schroff, F., et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR.
