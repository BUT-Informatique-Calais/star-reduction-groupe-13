from astropy.io import fits
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# -----------------------------
# Config
# -----------------------------
FITS_FILE = "./examples/test_M31_linear.fits"   # conseillé
OUT_DIR = "./results_m31"
os.makedirs(OUT_DIR, exist_ok=True)

# Réduction étoiles (image réduite)
KERNEL_SIZE = 5
ERODE_ITERS = 1

# "High-pass" pour isoler étoiles
BG_SIGMA = 20.0
DETAIL_DENOISE_SIGMA = 1.0

# Masque continu à partir de detail
MASK_BOOST = 3.5       # (plus grand = plus d'étoiles réduites)
MASK_MAX_STRENGTH = 0.90 # limite de force
MASK_SMOOTH_SIGMA = 1.5  # (adoucir transitions, anti-halos)

# Option: ignorer le très faible detail (bruit)
DETAIL_THRESHOLD = 8     # Monte si trop de bruit réduit.

# -----------------------------
# Helpers
# -----------------------------
def ensure_hwc(data: np.ndarray) -> np.ndarray:
    """ Assure que l'image est en format HWC (Height, Width, Channels)."""
    # (3,H,W) -> (H,W,3)
    if data.ndim == 3 and data.shape[0] == 3:
        return np.transpose(data, (1, 2, 0))
    return data

def norm01(x: np.ndarray) -> np.ndarray:
    """ Normalisation min-max entre 0 et 1, pour travailler avec OpenCV"""
    x = x.astype(np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def to_uint8_rgb_global(data_hwc: np.ndarray) -> np.ndarray:
    """
    Normalisation globale sur l'ensemble RGB pour conserver les ratios de couleur.
    """
    x01 = norm01(data_hwc)
    return (x01 * 255.0).astype(np.uint8)

# -----------------------------
# Chargement FITS
# -----------------------------
with fits.open(FITS_FILE) as hdul:
    hdul.info()
    data = hdul[0].data

data = ensure_hwc(data)
if data.ndim != 3 or data.shape[2] < 3:
    raise ValueError(f"Image RGB attendue, reçu shape={data.shape}")

data = data[:, :, :3]  # garde RGB

# -----------------------------
# Conversion en uint8
# -----------------------------
rgb_u8 = to_uint8_rgb_global(data)  # (H,W,3) RGB uint8
plt.imsave(os.path.join(OUT_DIR, "original.png"), rgb_u8)

bgr = cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)
gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
cv.imwrite(os.path.join(OUT_DIR, "gray.png"), gray)


# -----------------------------
# 1) Construction de l'image réduite (érosion par canal)
# -----------------------------

kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
reduced = bgr.copy()
for c in range(3):
    reduced[:, :, c] = cv.erode(bgr[:, :, c], kernel, iterations=ERODE_ITERS)

cv.imwrite(os.path.join(OUT_DIR, f"reduced_erode_k{KERNEL_SIZE}_it{ERODE_ITERS}.png"), reduced)



# -----------------------------
# 2) Calculer detail = gray - blurred(gray)
# -----------------------------

# Floute l'image afin de ne garder que les grandes structures (galaxie)
bg = cv.GaussianBlur(gray, (0, 0), sigmaX=BG_SIGMA)

# Retire la base floutée pour ne garder que le détail (étoiles + bruit)
detail = cv.subtract(gray, bg)

# Denoise léger du détail (adoucicement du bruit)
if DETAIL_DENOISE_SIGMA and DETAIL_DENOISE_SIGMA > 0:
    detail = cv.GaussianBlur(detail, (0, 0), sigmaX=DETAIL_DENOISE_SIGMA)

cv.imwrite(os.path.join(OUT_DIR, "bg.png"), bg)
cv.imwrite(os.path.join(OUT_DIR, "detail.png"), detail)

# Suppression du très faible détail (bruit)
if DETAIL_THRESHOLD > 0:
    _, detail_thr = cv.threshold(detail, DETAIL_THRESHOLD, 255, cv.THRESH_TOZERO)
else:
    detail_thr = detail.copy()

cv.imwrite(os.path.join(OUT_DIR, "detail_thr.png"), detail_thr)

# -----------------------------
# 3) Construciton du mask continu M depuis detail
# -----------------------------

#Conversion en float [0,1]
M = detail_thr.astype(np.float32) / 255.0

# Amplification du masque (plus d'étoiles réduites) et limitation
M = np.clip(M * MASK_BOOST, 0.0, MASK_MAX_STRENGTH)

# Adoucir transitions (anti-halos)
M = cv.GaussianBlur(M, (0, 0), sigmaX=MASK_SMOOTH_SIGMA)

cv.imwrite(os.path.join(OUT_DIR, "mask_continu.png"), (M * 255).astype(np.uint8))

# -----------------------------
# 4) Mélange de couleur
# final = M*reduced + (1-M)*original
# -----------------------------
I = bgr.astype(np.float32)
R = reduced.astype(np.float32)
M3 = M[:, :, None]

final_bgr = (M3 * R) + ((1.0 - M3) * I)
final_bgr = np.clip(final_bgr, 0, 255).astype(np.uint8)
cv.imwrite(os.path.join(OUT_DIR, "final_bgr.png"), final_bgr)

final_rgb = cv.cvtColor(final_bgr, cv.COLOR_BGR2RGB)
plt.imsave(os.path.join(OUT_DIR, "final_star_reduced.png"), final_rgb)

print("✅ Terminé (M31 - masque continu). Résultats :", OUT_DIR)
print("- original.png")
print("- bg.png / detail.png / detail_thr.png")
print("- mask_continu.png")
print("- reduced_erode_*.png")
print("- final_star_reduced.png")
