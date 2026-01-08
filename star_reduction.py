from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# -----------------------------
# Config
# -----------------------------
fits_file = "./examples/test_M31_linear.fits"


# 1) Érosion
KERNEL_SIZE = 5
ERODE_ITERS = 2

# 2) Masque étoiles (Top-Hat)
TOPHAT_SIZE = 21          
MEDIAN_SIZE = 5          
PERCENTILE = 99.4        

# Filtre composantes
MIN_AREA = 3
MAX_AREA = 600            
MAX_DIM  = 26             
MAX_ELONG = 2.2           

# Nettoyage/feather du masque
OPEN_ITERS = 1
DILATE_ITERS = 1          
BLUR_SIGMA = 1.6          
MASK_MAX = 0.90           
GAMMA = 0.70              

# -----------------------------
# Helpers
# -----------------------------
def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - mn) / (mx - mn + 1e-12)

def save_u8(name: str, img_u8: np.ndarray) -> None:
    cv.imwrite(os.path.join(OUT_DIR, name), img_u8)

# -----------------------------
# Load FITS -> image uint8 + gray (ROBUSTE)
# -----------------------------
with fits.open(fits_file) as hdul:
    data = hdul[0].data

# Cas RGB FITS: (3,H,W) -> (H,W,3)
if data.ndim == 3 and data.shape[0] == 3:
    data = np.transpose(data, (1, 2, 0))

# --- RGB ---
if data.ndim == 3:
    OUT_DIR = "./results_m31"
    data01 = norm01(data)
    plt.imsave(os.path.join(OUT_DIR, "original.png"), data01)

    image_rgb = (data01 * 255).astype(np.uint8)
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

# --- MONO ---
else:
    OUT_DIR = "./results_horsehead"
    data01 = norm01(data)
    plt.imsave(os.path.join(OUT_DIR, "original.png"), data01, cmap="gray")

    gray = (data01 * 255).astype(np.uint8)

    image_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================
# 1) Erosion
# =========================================
kernel_erode = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
Ierode = cv.erode(image_bgr, kernel_erode, iterations=ERODE_ITERS)


# =========================================
# 2) M = masque d’étoiles + flou
# =========================================

# --- Top-Hat : met en évidence les petites structures brillantes (principalement les étoiles)
k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TOPHAT_SIZE, TOPHAT_SIZE))
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, k)


# --- Filtre médian pour réduire le bruit
if MEDIAN_SIZE and MEDIAN_SIZE >= 3:
    tophat_f = cv.medianBlur(tophat, MEDIAN_SIZE)
else:
    tophat_f = tophat.copy()

# --- Seuil automatique basé sur un percentile des valeurs du Top-Hat
thr = float(np.percentile(tophat_f, PERCENTILE))
thr = max(thr, 8.0)  # sécurité

# --- Masque binaire initial
_, mask0 = cv.threshold(tophat_f, thr, 255, cv.THRESH_BINARY)

# Ouverture pour virer poussières de bruit
k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask0 = cv.morphologyEx(mask0, cv.MORPH_OPEN, k3, iterations=OPEN_ITERS)

# --- Filtrage composantes (garde ce qui ressemble à une étoile)
    # Regroupe les pixels blancs adjacents en "composantes connexes"
num, labels, stats, _ = cv.connectedComponentsWithStats(mask0, connectivity=8)
mask = np.zeros_like(mask0)

for i in range(1, num):
    area = stats[i, cv.CC_STAT_AREA]
    w = stats[i, cv.CC_STAT_WIDTH]
    h = stats[i, cv.CC_STAT_HEIGHT]

    # Critère de correspondance à une étoile sur la taille
    if area < MIN_AREA or area > MAX_AREA:
        continue

    # Mesures géométriques de la composante (bbox + élongation)
    max_dim = max(w, h)
    min_dim = min(w, h)
    elong = max_dim / (min_dim + 1e-6)

    # Critère de correspondance à une étoile sur la forme
    if max_dim > MAX_DIM:
        continue
    
    # Critère de correspondance à une étoile sur l'élongation
    if elong > MAX_ELONG:
        continue
    # Garder les composantes valide dans le masque final
    mask[labels == i] = 255

# Couvrir le halo des étoiles
if DILATE_ITERS > 0:
    mask = cv.dilate(mask, k3, iterations=DILATE_ITERS)

# Flou gaussien
mask_blur = cv.GaussianBlur(mask, (0, 0), sigmaX=BLUR_SIGMA)

# Normalisation 
M = mask_blur.astype(np.float32) / 255.0
M = np.clip(M, 0.0, 1.0)
M = (M ** GAMMA)          # ajuste la courbe du masque
M = np.clip(M, 0.0, MASK_MAX)
save_u8("mask_strength.png", (M * 255).astype(np.uint8))

# =========================================
# ÉTAPE B : 3) Ifinal = (M*Ierode) + ((1-M)*Ioriginal)
# =========================================
I = image_bgr.astype(np.float32)
E = Ierode.astype(np.float32)


if I.ndim == 2:
    Ifinal = (M * E) + ((1.0 - M) * I)
else:
    Ifinal = (M[:, :, None] * E) + ((1.0 - M)[:, :, None] * I)

Ifinal = np.clip(Ifinal, 0, 255).astype(np.uint8)
save_u8("final_star_reduced.png", Ifinal)

print("✅ Images générées")
print(f"   Seuil auto utilisé (percentile {PERCENTILE}) = {thr:.2f}")
