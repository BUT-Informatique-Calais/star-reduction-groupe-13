from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# -----------------------------
# Config
# -----------------------------
fits_file = "./examples/HorseHead.fits"
OUT_DIR = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Érosion
KERNEL_SIZE = 5
ERODE_ITERS = 2

# 2) Masque étoiles (Top-Hat)
TOPHAT_SIZE = 21          # plus petit => moins de texture nébuleuse dans le masque
MEDIAN_SIZE = 5           # aide beaucoup contre le bruit qui devient "étoiles"
PERCENTILE = 99.4         # seuil auto: plus bas => + d’étoiles

# Filtre composantes (évite texture nébuleuse)
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
# Load FITS -> image uint8 + gray
# -----------------------------
with fits.open(fits_file) as hdul:
    data = hdul[0].data

# Cas RGB FITS: (3,H,W) -> (H,W,3)
if data.ndim == 3 and data.shape[0] == 3:
    data = np.transpose(data, (1, 2, 0))

if data.ndim == 3:
    data01 = norm01(data)
    plt.imsave(os.path.join(OUT_DIR, "original.png"), data01)
    image = (data01 * 255).astype(np.uint8)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
else:
    data01 = norm01(data)
    plt.imsave(os.path.join(OUT_DIR, "original.png"), data01, cmap="gray")
    image = (data01 * 255).astype(np.uint8)
    gray = image.copy()

save_u8("gray.png", gray)

# =========================================
# 1) Erosion
# =========================================
kernel_erode = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
Ierode = cv.erode(image, kernel_erode, iterations=ERODE_ITERS)
save_u8("eroded.png", Ierode)

# =========================================
# 2) M = masque d’étoiles + flou
# =========================================
k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (TOPHAT_SIZE, TOPHAT_SIZE))
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, k)
save_u8("tophat.png", tophat)

if MEDIAN_SIZE and MEDIAN_SIZE >= 3:
    tophat_f = cv.medianBlur(tophat, MEDIAN_SIZE)
else:
    tophat_f = tophat.copy()
save_u8("tophat_med.png", tophat_f)

# --- Seuil auto
thr = float(np.percentile(tophat_f, PERCENTILE))
thr = max(thr, 8.0)  # sécurité
_, mask0 = cv.threshold(tophat_f, thr, 255, cv.THRESH_BINARY)

# Ouverture pour virer poussières de bruit
k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask0 = cv.morphologyEx(mask0, cv.MORPH_OPEN, k3, iterations=OPEN_ITERS)

# --- Filtrage composantes (garde ce qui ressemble à une étoile)
num, labels, stats, _ = cv.connectedComponentsWithStats(mask0, connectivity=8)
mask = np.zeros_like(mask0)

for i in range(1, num):
    area = stats[i, cv.CC_STAT_AREA]
    w = stats[i, cv.CC_STAT_WIDTH]
    h = stats[i, cv.CC_STAT_HEIGHT]

    if area < MIN_AREA or area > MAX_AREA:
        continue

    max_dim = max(w, h)
    min_dim = min(w, h)
    elong = max_dim / (min_dim + 1e-6)

    if max_dim > MAX_DIM:
        continue
    if elong > MAX_ELONG:
        continue

    mask[labels == i] = 255

save_u8("mask_raw.png", mask)

# Halo léger (optionnel)
if DILATE_ITERS > 0:
    mask = cv.dilate(mask, k3, iterations=DILATE_ITERS)

# Lissage des bords
mask_blur = cv.GaussianBlur(mask, (0, 0), sigmaX=BLUR_SIGMA)
save_u8("mask_blur.png", mask_blur)

M = mask_blur.astype(np.float32) / 255.0
M = np.clip(M, 0.0, 1.0)
M = (M ** GAMMA)          # renforce le centre des étoiles sans étaler partout
M = np.clip(M, 0.0, MASK_MAX)
save_u8("mask_strength.png", (M * 255).astype(np.uint8))

# =========================================
# ÉTAPE B : 3) Ifinal = (M*Ierode) + ((1-M)*Ioriginal)
# =========================================
I = image.astype(np.float32)
E = Ierode.astype(np.float32)

if I.ndim == 2:
    Ifinal = (M * E) + ((1.0 - M) * I)
else:
    Ifinal = (M[:, :, None] * E) + ((1.0 - M)[:, :, None] * I)

Ifinal = np.clip(Ifinal, 0, 255).astype(np.uint8)
save_u8("final_star_reduced.png", Ifinal)

print("✅ Étape B terminée (seuil auto + masque propre)")
print(f"   Seuil auto utilisé (percentile {PERCENTILE}) = {thr:.2f}")
print("   ./results : tophat.png, tophat_med.png, mask_raw.png, mask_strength.png, final_star_reduced.png")
