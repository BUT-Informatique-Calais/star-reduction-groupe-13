import numpy as np
import cv2 as cv
import os
from astropy.io import fits

# Configuration
FITS_FILE = "./examples/test_M31_linear.fits"
OUT_DIR = "./results_m31_sans_etoile" 
os.makedirs(OUT_DIR, exist_ok=True)

STAR_SIZE_LIMIT = 7      
MASK_DILATE = 2          
REDUCTION_STRENGTH = 0.8 
MEDIAN_SIZE = 5          

# Helpers
def ensure_hwc(data):
    if data.ndim == 3 and data.shape[0] == 3:
        return np.transpose(data, (1, 2, 0))
    return data

def to_uint16(data):
    """ Convertit en 16-bits (0 à 65535) pour une haute précision """
    mn, mx = np.nanmin(data), np.nanmax(data)
    if mx - mn < 1e-10: return np.zeros(data.shape, dtype=np.uint16)
    return (65535 * (data - mn) / (mx - mn)).astype(np.uint16)


# 1) Chargement et Préparation
with fits.open(FITS_FILE) as hdul:
    raw_data = hdul[0].data
    data_hwc = ensure_hwc(raw_data)
    img_rgb = data_hwc[:, :, :3]

# Conversion en 16 bits
img_16bit = to_uint16(img_rgb)
bgr = cv.cvtColor(img_16bit, cv.COLOR_RGB2BGR)
gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)


# 2) Détection

# Le TopHat
kernel_detect = cv.getStructuringElement(cv.MORPH_ELLIPSE, (STAR_SIZE_LIMIT, STAR_SIZE_LIMIT))
stars_only = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel_detect)


# 3) Raffinement du Masque

threshold_val = 2500 
_, mask = cv.threshold(stars_only, threshold_val, 65535, cv.THRESH_BINARY)

kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask = cv.dilate(mask, kernel_dilate, iterations=MASK_DILATE)

mask_smooth = cv.GaussianBlur(mask, (15, 15), 0)
mask_final = (mask_smooth.astype(np.float32) / 65535.0) * REDUCTION_STRENGTH


# 4) Remplacement
img_reduced = cv.medianBlur(bgr, MEDIAN_SIZE)

# 5) Fusion Finale

m = mask_final[:, :, None]
# On fait le calcul de mélange en float32 pour la précision
final_float = (m * img_reduced.astype(np.float32)) + ((1 - m) * bgr.astype(np.float32))

# On repasse en 16 bits proprement pour la sauvegarde
final_16bit = np.clip(final_float, 0, 65535).astype(np.uint16)


# Sauvegardes

cv.imwrite(os.path.join(OUT_DIR, "01_mask_pro_16bit.png"), (mask_final * 65535).astype(np.uint16))
cv.imwrite(os.path.join(OUT_DIR, "03_final_pro_16bit.png"), final_16bit)

print(f" Traitement 16-bits terminé. Dossier : {OUT_DIR}")