import numpy as np

def compute_ndvi(nir_band, red_band):
    # NDVI = (NIR - RED) / (NIR + RED)
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    return np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

def compute_nbr(nir_band, swir_band):
    # NBR = (NIR - SWIR) / (NIR + SWIR)
    nir = nir_band.astype(float)
    swir = swir_band.astype(float)
    return np.where((nir + swir) == 0, 0, (nir - swir) / (nir + swir))

def dnbr(pre_nbr, post_nbr):
    # dNBR = pre-fire NBR - post-fire NBR
    # > 0.27 = high severity burn
    return pre_nbr - post_nbr

