import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CBERS4MUXDataset(Dataset):
    """
    Dataset para imagens CBERS-4 MUX.
    - Carrega as 4 bandas originais (B, G, R, NIR) com rasterio.
    - Opcionalmente, calcula e adiciona índices espectrais como canais extras.
    """
    def __init__(self, red_image_paths, green_image_paths, blue_image_paths, nir_image_paths, mask_paths, 
                 indices_to_add=None, transform=None):
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
                ndwi_green = (image_green - image_nir) / (image_green + image_nir)
                all_bands.append(np.nan_to_num(ndwi_green)) 
                
            if 'GNDVI' in self.indices_to_add:
                gndvi = (image_nir - image_green) / (image_nir + image_green)
                all_bands.append(np.nan_to_num(gndvi))

        combined_image = np.stack(all_bands, axis=-1)
        combined_image[:, :, :4] = combined_image[:, :, :4] / 4095.0
        
        mask_rgba = Image.open(self.mask_paths[idx]).convert("RGBA")
        _, _, _, alpha = mask_rgba.split()
        mask = np.array(alpha)
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=combined_image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, mask.unsqueeze(0)

class CBERS4MUXDatasetWHidrography(Dataset):
    """
    Dataset para imagens CBERS-4 MUX.
    - Carrega as 4 bandas originais (B, G, R, NIR) com rasterio.
    - Opcionalmente, calcula e adiciona índices espectrais como canais extras.
    """
    def __init__(self, red_image_paths, green_image_paths, blue_image_paths, nir_image_paths, mask_paths, hidrography_paths, 
                 indices_to_add=None, transform=None):
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
        
        all_bands = [image_blue, image_green, image_red, image_nir, hidrography]

        if self.indices_to_add:
            np.seterr(divide='ignore', invalid='ignore')
                                                                            
            if 'NDVI' in self.indices_to_add:
                ndvi = (image_nir - image_red) / (image_nir + image_red)
                all_bands.append(np.nan_to_num(ndvi))
            
            if 'NDWI' in self.indices_to_add:
                ndwi_green = (image_green - image_nir) / (image_green + image_nir)
                all_bands.append(np.nan_to_num(ndwi_green)) 
                
            if 'GNDVI' in self.indices_to_add:
                gndvi = (image_nir - image_green) / (image_nir + image_green)
                all_bands.append(np.nan_to_num(gndvi))

        combined_image = np.stack(all_bands, axis=-1)
        combined_image[:, :, :4] = combined_image[:, :, :4] / 4095.0
        
        mask_rgba = Image.open(self.mask_paths[idx]).convert("RGBA")
        _, _, _, alpha = mask_rgba.split()
        mask = np.array(alpha)
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=combined_image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, mask.unsqueeze(0)