import numpy as np
import cv2 as cv
import os
from astropy.io import fits


# Configuration
FITS_FILE = "./examples/test_M31_linear.fits"
OUT_DIR = "./result_m31_with_star"  # On crée s'il n'existe pas le fichier ou les images sont stockés.
os.makedirs(OUT_DIR, exist_ok=True)

# Paramètres affinés pour M31
STAR_SIZE_LIMIT = 4       # Pluôt petit pour esquiver les nuages de poussières
REDUCTION_STRENGTH = 0.6  # 60% de réduction 
MEDIAN_SIZE = 3           # Lissage fin


# Fonctions Helpers
def ensure_hwc(data):
    if data.ndim == 3 and data.shape[0] == 3:
        return np.transpose(data, (1, 2, 0))
    return data

def arcsinh_stretch(data, softness=0.1):
    return np.arcsinh(data / softness) / np.arcsinh(1.0 / softness)

with fits.open(FITS_FILE) as hdul:
    #float 64
    raw_data = hdul[0].data.astype(np.float64)
    data = ensure_hwc(raw_data)
    # On s'assure d'avoir 3 canaux RGB
    if data.shape[2] > 3:
        data = data[:, :, :3]


# 2) Masque

# On crée une copie normalisée juste pour calculer le masque.
gray = np.mean(data, axis=2)
gray_norm = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))

# Top-Hat : Isole les étoiles ponctuelles
kernel_detect = cv.getStructuringElement(cv.MORPH_ELLIPSE, (STAR_SIZE_LIMIT, STAR_SIZE_LIMIT))
stars_only = cv.morphologyEx(gray_norm.astype(np.float32), cv.MORPH_TOPHAT, kernel_detect)

# Masque Binaire
_, mask = cv.threshold(stars_only, 0.02, 1.0, cv.THRESH_BINARY)

# Dilatation et Adoucissement
mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
mask = cv.GaussianBlur(mask, (3, 3), 0)
mask = mask[:, :, None] # Expansion pour le RGB


# 3) Traitement

# On applique le filtre médian sur les données float64
img_reduced = cv.medianBlur(data.astype(np.float32), MEDIAN_SIZE).astype(np.float64)

# Fusion Linéaire
final_scientific = (data * (1.0 - (mask * REDUCTION_STRENGTH))) + \
                   (img_reduced * (mask * REDUCTION_STRENGTH))


# 4) Sauvegardes


# Sauvegarde FITS
# On remet les axes en (C, H, W) pour le standard FITS
hdu_save = fits.PrimaryHDU(np.transpose(final_scientific, (2, 0, 1)))
hdu_save.writeto(os.path.join(OUT_DIR, "M31_reduced_scientific.fits"), overwrite=True)

# Sauvegarde PNG
visual = arcsinh_stretch(final_scientific, softness=0.005)
black_point = np.percentile(visual, 5)
visual = np.clip((visual - black_point) / (1.0 - black_point), 0, 1)
# Conversion 8-bit pour affichage
visual_bgr = cv.cvtColor((visual * 255).astype(np.uint8), cv.COLOR_RGB2BGR)

cv.imwrite(os.path.join(OUT_DIR, "M31_visualisation_belle.png"), visual_bgr)
cv.imwrite(os.path.join(OUT_DIR, "M31_masque_check.png"), (mask * 255).astype(np.uint8))

print(f" Terminé. Fichiers sauvegardés dans : {OUT_DIR}")