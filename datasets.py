import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def _load_binary_mask(mask_path: str, threshold: int = 128, invert='auto', target_color=None) -> np.ndarray:
    """Load a mask as binary array, handling grayscale, RGB and RGBA inputs.
    
    Args:
        mask_path: Path to mask file
        threshold: Threshold for grayscale masks
        invert: 'auto', True, or False
        target_color: Tuple (R, G, B) to extract specific color from RGB/RGBA masks
    """
    with Image.open(mask_path) as mask_img:
        mask_np = np.array(mask_img)

    if mask_np.ndim == 2:
        mask_base = mask_np
    elif mask_np.ndim == 3:
        if target_color is not None:
            target = np.array(target_color, dtype=mask_np.dtype)
            if mask_np.shape[2] == 4:
                rgb = mask_np[:, :, :3]
                alpha = mask_np[:, :, 3]
                color_match = np.all(rgb == target, axis=2)
                mask_base = (color_match & (alpha > 0)).astype(np.float32)
            else:
                color_match = np.all(mask_np[:, :, :3] == target, axis=2)
                mask_base = color_match.astype(np.float32)
        else:
            if mask_np.shape[2] == 4:
                alpha = mask_np[:, :, 3]
                rgb = mask_np[:, :, :3]

                if np.unique(alpha).size > 1:
                    mask_base = alpha
                else:
                    mask_base = np.mean(rgb, axis=2)
            else:
                mask_base = np.mean(mask_np[:, :, :3], axis=2)
    else:
        raise ValueError(
            f"Formato de máscara não suportado em {mask_path}: shape={mask_np.shape}")

    if target_color is None:
        mask_binary = (mask_base > threshold).astype(np.float32)
    else:
        mask_binary = mask_base

    # Mixed datasets may use opposite conventions (foreground=white or foreground=black).
    # In auto mode, keep foreground as the minority class to stabilize training.
    if invert == 'auto':
        if mask_binary.mean() > 0.5:
            mask_binary = 1.0 - mask_binary
    elif invert:
        mask_binary = 1.0 - mask_binary

    return mask_binary


class CBERS4MUXDataset(Dataset):
    """
    Dataset para imagens CBERS-4 MUX.
    - Carrega as 4 bandas originais (B, G, R, NIR) com rasterio.
    - Opcionalmente, calcula e adiciona índices espectrais como canais extras.
    """

    def __init__(self, red_image_paths, green_image_paths, blue_image_paths, nir_image_paths, mask_paths,
                 indices_to_add=None, transform=None, mask_invert='auto'):
        """
        Args:
            ... (caminhos para as imagens e máscaras)
            indices_to_add (list, optional): Lista de strings com os nomes dos índices a serem adicionados 
                                            como canais. Ex: ['NDVI', 'EVI']. Default é None.
            transform (callable, optional): Transformações/augmentations a serem aplicadas.
        """
        self.red_image_paths = red_image_paths
        self.green_image_paths = green_image_paths
        self.blue_image_paths = blue_image_paths
        self.nir_image_paths = nir_image_paths
        self.mask_paths = mask_paths
        self.indices_to_add = indices_to_add if indices_to_add is not None else []
        self.transform = transform
        self.mask_invert = mask_invert

    def __len__(self):
        return len(self.red_image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.red_image_paths[idx]) as src:
            image_red = src.read(1).astype(np.float32)
        with rasterio.open(self.green_image_paths[idx]) as src:
            image_green = src.read(1).astype(np.float32)
        with rasterio.open(self.blue_image_paths[idx]) as src:
            image_blue = src.read(1).astype(np.float32)
        with rasterio.open(self.nir_image_paths[idx]) as src:
            image_nir = src.read(1).astype(np.float32)

        all_bands = [image_blue, image_green, image_red, image_nir]

        if self.indices_to_add:
            np.seterr(divide='ignore', invalid='ignore')

            if 'NDVI' in self.indices_to_add:
                ndvi = (image_nir - image_red) / (image_nir + image_red)
                all_bands.append(np.nan_to_num(ndvi))

            if 'NDWI' in self.indices_to_add:
                ndwi_green = (image_green - image_nir) / \
                    (image_green + image_nir)
                all_bands.append(np.nan_to_num(ndwi_green))

            if 'GNDVI' in self.indices_to_add:
                gndvi = (image_nir - image_green) / (image_nir + image_green)
                all_bands.append(np.nan_to_num(gndvi))

        combined_image = np.stack(all_bands, axis=-1)
        combined_image[:, :, :4] = combined_image[:, :, :4] / 4095.0

        mask = _load_binary_mask(self.mask_paths[idx], invert=self.mask_invert, target_color=(133, 196, 221))

        if self.transform:
            transformed = self.transform(image=combined_image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(combined_image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        return image, mask.unsqueeze(0)


class CBERS4MUXDatasetWHidrography(Dataset):
    """
    Dataset para imagens CBERS-4 MUX.
    - Carrega as 4 bandas originais (B, G, R, NIR) com rasterio.
    - Opcionalmente, calcula e adiciona índices espectrais como canais extras.
    """

    def __init__(self, red_image_paths, green_image_paths, blue_image_paths, nir_image_paths, mask_paths, hidrography_paths,
                 indices_to_add=None, transform=None, mask_invert='auto'):
        """
        Args:
            ... (caminhos para as imagens e máscaras)
            indices_to_add (list, optional): Lista de strings com os nomes dos índices a serem adicionados 
                                            como canais. Ex: ['NDVI', 'EVI']. Default é None.
            transform (callable, optional): Transformações/augmentations a serem aplicadas.
        """
        self.red_image_paths = red_image_paths
        self.green_image_paths = green_image_paths
        self.blue_image_paths = blue_image_paths
        self.nir_image_paths = nir_image_paths
        self.mask_paths = mask_paths
        self.hidrography_paths = hidrography_paths
        self.indices_to_add = indices_to_add if indices_to_add is not None else []
        self.transform = transform
        self.mask_invert = mask_invert

    def __len__(self):
        return len(self.red_image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.red_image_paths[idx]) as src:
            image_red = src.read(1).astype(np.float32)
        with rasterio.open(self.green_image_paths[idx]) as src:
            image_green = src.read(1).astype(np.float32)
        with rasterio.open(self.blue_image_paths[idx]) as src:
            image_blue = src.read(1).astype(np.float32)
        with rasterio.open(self.nir_image_paths[idx]) as src:
            image_nir = src.read(1).astype(np.float32)
        with rasterio.open(self.hidrography_paths[idx]) as src:
            hidrography = src.read(1).astype(np.float32)

        all_bands = [image_blue, image_green,
                     image_red, image_nir, hidrography]

        if self.indices_to_add:
            np.seterr(divide='ignore', invalid='ignore')

            if 'NDVI' in self.indices_to_add:
                ndvi = (image_nir - image_red) / (image_nir + image_red)
                all_bands.append(np.nan_to_num(ndvi))

            if 'NDWI' in self.indices_to_add:
                ndwi_green = (image_green - image_nir) / \
                    (image_green + image_nir)
                all_bands.append(np.nan_to_num(ndwi_green))

            if 'GNDVI' in self.indices_to_add:
                gndvi = (image_nir - image_green) / (image_nir + image_green)
                all_bands.append(np.nan_to_num(gndvi))

        combined_image = np.stack(all_bands, axis=-1)
        combined_image[:, :, :4] = combined_image[:, :, :4] / 4095.0

        mask = _load_binary_mask(self.mask_paths[idx], invert=self.mask_invert, target_color=(133, 196, 221))

        if self.transform:
            transformed = self.transform(image=combined_image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(combined_image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        return image, mask.unsqueeze(0)


class CBERS4MUXSiameseDataset(Dataset):
    """
    Dataset para rede siamesa. Retorna 3 tensores de entrada separados:
      - rgb: (3, H, W) — canais B, G, R
      - spectral: (7, H, W) — B, G, R, NIR, NDVI, NDWI, GNDVI
      - hydro: (1, H, W) — mapa de hidrografia
      - mask: (1, H, W) — ground truth binario

    Sem crop (preserva tamanho original 256x256).
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

        # Ordem dos canais: [B, G, R, NIR, HYDRO, NDVI, NDWI, GNDVI]
        rgb = full_image[:3, :, :]
        hydro = full_image[4:5, :, :]
        spectral = torch.cat(
            [full_image[:4, :, :], full_image[5:, :, :]], dim=0)

        return rgb, spectral, hydro, mask
