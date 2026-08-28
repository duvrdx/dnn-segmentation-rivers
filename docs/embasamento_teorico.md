# Embasamento Teórico — Siamese Cross-Attention para Segmentação de Rios

## Referências Bibliográficas por Componente Arquitetural

---

## 1. Detecção de Corpos d'Água em Sensoriamento Remoto

### Índices Espectrais para Água

O uso de índices espectrais é a abordagem clássica e mais estabelecida para detecção de água em imagens de satélite:

| Índice | Fórmula | Referência | Propósito |
|---|---|---|---|
| **NDWI** | `(Green - NIR) / (Green + NIR)` | McFeeters (1996) | Realce de corpos d'água abertos. Usa forte absorção no NIR e reflectância no verde. Citado >6.700 vezes. |
| **MNDWI** | `(Green - SWIR) / (Green + SWIR)` | Xu (2006) | Modificação do NDWI para suprimir ruído de áreas urbanas e solo. |
| **NDVI** | `(NIR - Red) / (NIR + Red)` | Rouse et al. (1973) | Índice de vegetação; água apresenta valores negativos. |
| **GNDVI** | `(NIR - Green) / (NIR + Green)` | Gitelson et al. (1996) | Mais sensível à clorofila que NDVI; útil para discriminar água de vegetação aquática. |

**Referências:**
- McFeeters, S.K. (1996). *The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.* International Journal of Remote Sensing, 17, 1425-1432. [DOI: 10.1080/01431169608948714](https://doi.org/10.1080/01431169608948714)
- Xu, H. (2006). *Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery.* International Journal of Remote Sensing, 27(14), 3025-3033. [DOI: 10.1080/01431160600589179](https://doi.org/10.1080/01431160600589179)

### O Satélite CBERS-4 MUX

O projeto utiliza imagens do satélite CBERS-4 (China-Brazil Earth Resources Satellite), sensor MUX, com 4 bandas espectrais de 20m de resolução:
- Banda 1: Blue (450-520 nm)
- Banda 2: Green (520-590 nm)
- Banda 3: Red (630-690 nm)
- Banda 4: Near-Infrared (770-890 nm)

A ausência de bandas SWIR limita o uso do MNDWI, mas permite NDVI, NDWI (Green/NIR) e GNDVI.

---

## 2. Arquiteturas de Segmentação Semântica

### 2.1 U-Net

**Ronneberger, O., Fischer, P., & Brox, T. (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.

Arquitetura encoder-decoder com skip connections que preservam detalhes espaciais. Amplamente adotada em sensoriamento remoto por sua eficiência com poucos dados. Já implementada no projeto como baseline (`unet.py`).

### 2.2 DeepLabV3+

**Chen, L.C., et al. (2018).** *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV 2018.

Combina Atrous Spatial Pyramid Pooling (ASPP) para capturar contexto multi-escala com um decoder simples que refina bordas. É a arquitetura de melhor performance no projeto até o momento (**0.8728 IoU** no rio Doce com guidance de hidrografia).

### 2.3 SegFormer

**Xie, E., et al. (2021).** *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021. [arXiv: 2105.15203](https://arxiv.org/abs/2105.15203)

- Encoder transformer hierárquico (MiT) que produz features multi-escala
- Não precisa de positional encoding → robusto a mudanças de resolução
- Decoder MLP leve que agrega informações de múltiplas camadas
- Backbone `mit_b5` já testado no projeto (0.8173 IoU)

---

## 3. Redes Siamesas e Multi-Branch

### 3.1 Conceito Original

**Bromley, J., et al. (1993).** *Signature Verification using a "Siamese" Time Delay Neural Network.* NIPS 1993.

Redes siamesas clássicas utilizam **pesos compartilhados** entre dois branches para aprender representações comparáveis. No presente projeto, adaptamos o conceito para **3 branches com entradas heterogêneas** (RGB, espectro completo, hidrografia), onde cada branch processa uma modalidade diferente com pesos *não compartilhados*.

### 3.2 Two-Stream Networks

**Simonyan, K. & Zisserman, A. (2014).** *Two-Stream Convolutional Networks for Action Recognition in Videos.* NeurIPS 2014.

Inspiração direta para o design de múltiplos streams processando diferentes modalidades de entrada (RGB + fluxo óptico no original; RGB + NIR+Índices + Hidrografia no presente projeto).

### 3.3 Fusão Multi-Modal em Sensoriamento Remoto

**Tong, Q., et al. (2024).** *CrossFormer Embedding DeepLabv3+ for Remote Sensing Images Semantic Segmentation.* Computers, Materials & Continua, 79(1), 1353-1375. [DOI: 10.32604/cmc.2024.049187](https://doi.org/10.32604/cmc.2024.049187)

Propõe CrossFormer + DeepLabV3+ com mecanismo de self-attention cross-região para features multi-escala, melhorando segmentação de bordas e objetos pequenos em imagens de alta resolução.

**Li, et al. (2025).** *MMA-Net: A Semantic Segmentation Network for High-Resolution Remote Sensing Images Based on Multimodal Fusion and Multi-Scale Multi-Attention Mechanisms.* Remote Sensing, 17(21), 3572. [DOI: 10.3390/rs17213572](https://www.mdpi.com/2072-4292/17/21/3572)

Propõe fusão cross-layer multimodal com cross-attention + self-attention, atingindo 88.74% mIoU no Potsdam. Valida que a fusão atenta entre modalidades supera concatenação simples.

---

## 4. Mecanismos de Atenção e Fusão Cross-Modal

### 4.1 Transformer / Attention

**Vaswani, A., et al. (2017).** *Attention Is All You Need.* NeurIPS 2017.

Mecanismo de atenção escalonada por produto escalar (scaled dot-product attention):
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Cross-attention** ocorre quando Q e K/V vêm de fontes diferentes (ex: Q do RGB, K/V do NIR). Permite que um branch "consulte" o outro, modelando relações espaciais entre modalidades.

### 4.2 CrossViT

**Chen, C.F., et al. (2021).** *CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification.* ICCV 2021.

Introduz cross-attention entre patches de diferentes escalas. Inspiração para a fusão bidirecional RGB ↔ NIR no presente projeto.

### 4.3 FiLM — Feature-wise Linear Modulation

**Perez, E., et al. (2018).** *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. [arXiv: 1709.07871](https://arxiv.org/abs/1709.07871)

Método de condicionamento que modula features através de transformação afim por canal:
```
FiLM(f) = γ * f + β
```
Onde γ e β são preditos a partir da informação de condicionamento.

**Por que FiLM para hidrografia?**
- A hidrografia é um **prior espacial binário** (rio/não-rio do IBGE)
- FiLM permite que atue como **condicionamento suave** (realce/supressão de canais)
- Muito mais leve que um encoder completo (usado na siamesa atual)

---

## 5. Fusão Multi-Escala

### 5.1 Feature Pyramid Networks

**Lin, T.Y., et al. (2017).** *Feature Pyramid Networks for Object Detection.* CVPR 2017.

Arquitetura que combina features de baixa resolução (semânticas) com alta resolução (detalhes espaciais) através de conexões laterais top-down. Aplicado diretamente aos níveis 2-5 dos encoders SMP.

### 5.2 Deep Supervision

**Lee, C.Y., et al. (2015).** *Deeply-Supervised Nets.* AISTATS 2015.

Adiciona supervisão em múltiplas escalas intermediárias, forçando camadas ocultas a aprender representações discriminativas. Benefícios:
- Convergência mais rápida
- Melhor fluxo de gradiente
- Representações multi-escala mais robustas

**Aplicação:** MSDS-UNet (Yang et al., 2021) — U-Net 3D com dupla via de deep supervision para segmentação de tumores. [DOI: 10.1016/j.compmedimag.2021.101957](https://doi.org/10.1016/j.compmedimag.2021.101957)

---

## 6. Triplet Loss e Mineração de Triplas

### 6.1 FaceNet

**Schroff, F., Kalenichenko, D., & Philbin, J. (2015).** *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR 2015.

Define a Triplet Loss:
```
L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + α)
```
Onde `a` = âncora, `p` = positivo (mesma classe), `n` = negativo (classe diferente), `α` = margem.

### 6.2 Batch-Hard Triplet Mining

**Hermans, A., Beyer, L., & Leibe, B. (2017).** *In Defense of the Triplet Loss for Person Re-Identification.* arXiv: 1703.07737.

Demonstra que a **amostragem aleatória** de triplas (usada na siamesa atual) é ineficiente — a perda triplet da siamesa atual é ~0.001 contra ~0.045 da seg_loss, indicando que não está aprendendo.

**Batch-hard mining:** para cada âncora, seleciona:
- **Hardest positive:** o positivo mais distante no batch
- **Hardest negative:** o negativo mais próximo no batch

Isso força o modelo a aprender separação onde é mais difícil, especialmente em bordas de rio, água rasa, e vegetação ripária.

---

## 7. K-Fold Cross-Validation

**Kohavi, R. (1995).** *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI 1995. [Stanford](http://robotics.stanford.edu/~ronnyk/accEst.pdf)

Estabelece que **10-fold stratified cross-validation** é o método mais robusto para:
- Estimativa de acurácia com menor viés
- Seleção de modelos com menor variância
- Comparação entre algoritmos

**Problema do método atual no projeto:**
```
for train_num in range(10):
    train_test_split(..., test_size=0.2, random_state=42)
```
Este loop treina 10 modelos na **mesma divisão** 80/20 — não há variabilidade entre folds. K-Fold (K=5) garante:
- Cada amostra é usada como validação exatamente uma vez
- Relatório com **média ± desvio padrão** (intervalo de confiança real)
- Detecção de overfitting a um split específico

---

## 8. Mapas de Hidrografia como Priors Espaciais

O projeto utiliza mapas de hidrografia do **IBGE** (Instituto Brasileiro de Geografia e Estatística) como informação auxiliar. Estes mapas representam a rede de drenagem oficial do país e funcionam como um **prior espacial** binário indicando a presença esperada de corpos d'água.

A fusão com hidrografia provou ser o componente mais impactante nos experimentos existentes:
- DeepLabV3+ **com** guidance: 0.8728 IoU (Doce)
- DeepLabV3+ **sem** guidance: 0.7966 IoU (combinado)
- **Ganho de ~8-10 pontos percentuais** com a inclusão do prior

O presente trabalho propõe substituir a fusão pós-inferência (peso fixo da abordagem atual) por um **condicionamento aprendido** via FiLM, onde a rede decide quanto confiar na hidrografia em cada região da imagem.

---

## Resumo da Cadeia de Referências

```
Segmentação de Água em Satélite
├── Índices Espectrais: McFeeters (1996), Xu (2006)
├── Arquiteturas Base (validadas no projeto)
│   ├── U-Net: Ronneberger (2015)
│   ├── DeepLabV3+: Chen (2018) ← baseline mais forte (0.87 IoU)
│   └── SegFormer: Xie (2021) ← backbone mit_b5
│
├── Arquitetura Proposta (Siamese Cross-Attention)
│   ├── Multi-branch: Bromley (1993), Simonyan (2014)
│   ├── Cross-attention: Vaswani (2017), Chen (2021)
│   ├── FiLM conditioning: Perez (2018)
│   ├── Multi-scale FPN: Lin (2017)
│   ├── Deep supervision: Lee (2015)
│   └── Batch-hard triplet: Hermans (2017)
│
└── Metodologia Experimental
    ├── K-Fold CV: Kohavi (1995) ← K=5
    └── Métricas: IoU, WIoU (distância de borda)
```
