from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# -----------------------------
# Config
# -----------------------------
fits_file = "./examples/HorseHead.fits"
os.makedirs("./results", exist_ok=True)

KERNEL_SIZE = 5
ERODE_ITERS = 2
BLUR_SIGMA = 2.5

# -----------------------------
# Chargement FITS
# -----------------------------
hdul = fits.open(fits_file)
hdul.info()
data = hdul[0].data
hdul.close()

# -----------------------------
# Handle mono / color
# -----------------------------
if data.ndim == 3:
    # Channels-first → channels-last
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))

    # Normalize for display
    data_norm = (data - data.min()) / (data.max() - data.min())
    plt.imsave("./results/original.png", data_norm)

    # Convert to uint8 for OpenCV
    image = np.zeros_like(data, dtype=np.uint8)
    for i in range(data.shape[2]):
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) /
                          (channel.max() - channel.min()) * 255).astype(np.uint8)

    # Convert to grayscale for mask
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

else:
    # Monochrome
    data_norm = (data - data.min()) / (data.max() - data.min())
    plt.imsave("./results/original.png", data_norm, cmap="gray")

    image = (data_norm * 255).astype(np.uint8)
    gray = image.copy()

# -----------------------------
# 1) Image érodée
# -----------------------------
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
eroded = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=1)
cv.imwrite("./results/eroded.png", eroded)

# -----------------------------
# 2) Masque d’étoiles (simple)
# -----------------------------
_, mask = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)

# Nettoyage léger
kernel_small = np.ones((3, 3), np.uint8)
eroded = cv.erode(image, kernel, iterations=ERODE_ITERS)

cv.imwrite("./results/mask_raw.png", mask)

# -----------------------------
# 3) Flou du masque
# -----------------------------
mask = cv.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
mask_blur = cv.GaussianBlur(mask, (0,0), sigmaX=BLUR_SIGMA)
M = mask_blur.astype(np.float32) / 255.0

M = np.clip(M, 0.0, 0.75)


# -----------------------------
# 4) Fusion sélective
# -----------------------------
I = image.astype(np.float32)
Ie = eroded.astype(np.float32)

if I.ndim == 2:
    final = (M * Ie) + ((1 - M) * I)
else:
    final = (M[:, :, None] * Ie) + ((1 - M[:, :, None]) * I)

final = np.clip(final, 0, 255).astype(np.uint8)
cv.imwrite("./results/final_star_reduced.png", final)   

print("✅ Phase 2 terminée : réduction d’étoiles sélective")
