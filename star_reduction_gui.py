import os
import cv2 as cv
import numpy as np
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog


def choose_fits_file():
    """Ouvre une boîte de dialogue pour choisir un fichier FITS."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits;*.fts;*.fit"), ("All files", "*")])
    root.destroy()
    return path


def norm01(x: np.ndarray) -> np.ndarray:
    """Normalise min-max entre 0 et 1 (robuste aux NaN)."""
    x = x.astype(np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def ensure_hwc(data: np.ndarray) -> np.ndarray:
    """Transforme (3,H,W) -> (H,W,3) si nécessaire."""
    if data.ndim == 3 and data.shape[0] == 3:
        return np.transpose(data, (1, 2, 0))
    return data


def process_image(image_u8, kernel_size, thresh_sigma_mult, iterations, bg_sigma):
    """Applique le flux de réduction : soustraction du fond, seuillage, masque, érosion ciblée.

    Retourne : eroded (uint8), mask (uint8), background (uint8), stars (uint8), thresh (int)
    """
    # travailler sur la luminance
    if image_u8.ndim == 3:
        gray = cv.cvtColor(image_u8, cv.COLOR_BGR2GRAY)
    else:
        gray = image_u8.copy()

    # estimation du fond par un flou gaussien large (paramétrable)
    background = cv.GaussianBlur(gray, (0, 0), sigmaX=float(bg_sigma))

    # mettre en évidence les petites sources (étoiles)
    stars = cv.subtract(gray, background)
    stars = cv.normalize(stars, None, 0, 255, cv.NORM_MINMAX)
    stars = cv.medianBlur(stars, 3)

    mean_val = float(stars.mean())
    std_val = float(stars.std())
    thresh = int(mean_val + float(thresh_sigma_mult) * std_val)
    _, mask = cv.threshold(stars, thresh, 255, cv.THRESH_BINARY)

    # léger dilate pour couvrir le coeur
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.dilate(mask, k3, iterations=1)

    # érosion sur la zone de masque uniquement
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if image_u8.ndim == 3:
        eroded = image_u8.copy()
        for c in range(image_u8.shape[2]):
            ch = image_u8[:, :, c]
            ch_er = cv.erode(ch, k, iterations=iterations)
            eroded[:, :, c][mask > 0] = ch_er[mask > 0]
    else:
        ch_er = cv.erode(gray, k, iterations=iterations)
        eroded = gray.copy()
        eroded[mask > 0] = ch_er[mask > 0]

    # convertir background & stars en uint8 pour affichage
    bg_u8 = np.clip(background, 0, 255).astype(np.uint8)
    stars_u8 = np.clip(stars, 0, 255).astype(np.uint8)

    return eroded, mask, bg_u8, stars_u8, thresh


def to_display(img):
    """Convertit image en uint8 BGR pour affichage OpenCV."""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def main():
    print("Choisissez un fichier FITS à ouvrir...")
    fits_path = choose_fits_file()
    if not fits_path:
        print("Aucun fichier sélectionné. Fin.")
        return

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # lecture FITS
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    data = ensure_hwc(data)
    data01 = norm01(data)
    image_u8 = (data01 * 255).astype(np.uint8)

    # Si image couleur issue du FITS (RGB), convertir en BGR pour OpenCV
    if image_u8.ndim == 3 and image_u8.shape[2] == 3:
        try:
            image_u8 = cv.cvtColor(image_u8, cv.COLOR_RGB2BGR)
        except cv.error:
            image_u8 = image_u8[..., ::-1]

    # fenêtre GUI
    win = "Star Reduction GUI"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.imshow(win, np.zeros((10, 10, 3), dtype=np.uint8))
    cv.waitKey(1)

    # trackbars (toutes ASCII pour éviter soucis d'encodage)
    # Noyau : taille du noyau d'érosion (impair)
    cv.createTrackbar("Noyau", win, 3, 11, lambda v: None)
    # SeuilSigma : multiplicateur du sigma pour construire le masque (1..10)
    cv.createTrackbar("SeuilSigma", win, 4, 10, lambda v: None)
    # Iterations : nombre d'itérations d'érosion
    cv.createTrackbar("Iterations", win, 1, 3, lambda v: None)


    while True:
        # quitter si la fenêtre a été fermée
        try:
            vis = cv.getWindowProperty(win, cv.WND_PROP_VISIBLE)
        except cv.error:
            break
        if vis < 1:
            break

        k = cv.getTrackbarPos("Noyau", win)
        if k < 1:
            k = 1
            cv.setTrackbarPos("Noyau", win, k)
        if k % 2 == 0:
            k += 1
            if k > 11:
                k = 11
            cv.setTrackbarPos("Noyau", win, k)

        sigma_mult = max(1, cv.getTrackbarPos("SeuilSigma", win))
        iters = max(1, cv.getTrackbarPos("Iterations", win))
        # BG sigma fixe (non exposé dans l'interface)
        bg_sigma = 25

        eroded, mask, bg, stars, thr = process_image(image_u8, k, sigma_mult, iters, bg_sigma)

        disp_orig = to_display(image_u8)
        disp_eroded = to_display(eroded)
        disp_mask = to_display(mask)

        h, w = disp_orig.shape[:2]
        target_h = 600
        scale = target_h / max(h, 1)
        if scale < 1.0:
            new_w = int(w * scale)
            disp_orig = cv.resize(disp_orig, (new_w, target_h))
            disp_eroded = cv.resize(disp_eroded, (new_w, target_h))
            disp_mask = cv.resize(disp_mask, (new_w, target_h))

        combined = cv.hconcat([disp_orig, disp_eroded, disp_mask])

        # libellés (Original | Érodé | Masque)
        w1 = disp_orig.shape[1]
        w2 = disp_eroded.shape[1]
        cv.putText(combined, "Original", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(combined, "Erode", (w1 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(combined, "Masque", (w1 + w2 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
		
        cv.imshow(win, combined)

        key = cv.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('s'):
            base = os.path.splitext(os.path.basename(fits_path))[0]
            cv.imwrite(os.path.join(results_dir, f"{base}_eroded_k{k}_it{iters}_th{int(thr)}.png"), eroded)
            cv.imwrite(os.path.join(results_dir, f"{base}_mask.png"), mask)
            cv.imwrite(os.path.join(results_dir, f"{base}_bg.png"), bg)
            print("Images sauvegardées dans:", results_dir)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()


    """Ouvre une boîte de dialogue pour choisir un fichier FITS."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits;*.fts;*.fit"), ("All files", "*")])
    root.destroy()


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def ensure_hwc(data: np.ndarray) -> np.ndarray:
    # FITS RGB peut être (3,H,W)
    if data.ndim == 3 and data.shape[0] == 3:
        return np.transpose(data, (1, 2, 0))
    return data


def process_image(image_u8, kernel_size, thresh_sigma_mult, iterations):
    """Applique la réduction d'étoiles ciblée et retourne l'image résultante.

    - image_u8: uint8 HWC (ou 2D)
    - kernel_size: int (odd)
    - thresh_sigma_mult: float
    - iterations: int
    """
    # travailler sur luminance
    if image_u8.ndim == 3:
        gray = cv.cvtColor(image_u8, cv.COLOR_BGR2GRAY)
    else:
        gray = image_u8.copy()

    # fond lisse pour préserver la nébuleuse
    BG_SIGMA = 25.0
    background = cv.GaussianBlur(gray, (0, 0), sigmaX=BG_SIGMA)

    # met en évidence les étoiles
    stars = cv.subtract(gray, background)
    stars = cv.normalize(stars, None, 0, 255, cv.NORM_MINMAX)
    stars = cv.medianBlur(stars, 3)

    mean_val = float(stars.mean())
    std_val = float(stars.std())
    thresh = int(mean_val + thresh_sigma_mult * std_val)
    _, mask = cv.threshold(stars, thresh, 255, cv.THRESH_BINARY)

    # dilate pour couvrir le coeur
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.dilate(mask, k3, iterations=1)

    # érosion sur la zone de masque seulement
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if image_u8.ndim == 3:
        eroded = image_u8.copy()
        for c in range(image_u8.shape[2]):
            ch = image_u8[:, :, c]
            ch_er = cv.erode(ch, k, iterations=iterations)
            eroded[:, :, c][mask > 0] = ch_er[mask > 0]
    else:
        ch_er = cv.erode(gray, k, iterations=iterations)
        eroded = gray.copy()
        eroded[mask > 0] = ch_er[mask > 0]

    return eroded, mask, background, stars, thresh


def to_display(img):
    """Convertit image en uint8 BGR pour affichage OpenCV."""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def main():
    print("Choisissez un fichier FITS à ouvrir...")
    fits_path = choose_fits_file()
    if not fits_path:
        print("Aucun fichier sélectionné. Fin.")
        return

    # dossier de résultats
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Lecture FITS
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    data = ensure_hwc(data)
    data01 = norm01(data)
    image_u8 = (data01 * 255).astype(np.uint8)
    # Si image couleur (RGB provenant du FITS), convertir en BGR pour OpenCV
    if image_u8.ndim == 3 and image_u8.shape[2] == 3:
        try:
            image_u8 = cv.cvtColor(image_u8, cv.COLOR_RGB2BGR)
        except cv.error:
            # fallback: swap channels manually
            image_u8 = image_u8[..., ::-1]

    # fenêtre et sliders
    win = "Interface Utilisateur Star Reduction"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    # Initialiser la fenêtre (évite l'erreur NULL window sur Windows)
    cv.imshow(win, np.zeros((10, 10, 3), dtype=np.uint8))
    cv.waitKey(1)

    # trackbars: kernel size (1..11), thresh sigma mult (1..10), iters (1..3)
    cv.createTrackbar("Noyau", win, 3, 11, lambda v: None)  # valeur initiale 3
    cv.createTrackbar("SeuilSigma", win, 4, 10, lambda v: None)  # multiplier
    # utiliser un label ASCII pour éviter les problèmes d'encodage
    cv.createTrackbar("Iterations", win, 1, 3, lambda v: None)

    # fix: ensure Noyau odd and >=1
    while True:
        # Vérifier si la fenêtre est toujours visible; si l'utilisateur la ferme,
        # quitter proprement la boucle (ne pas la recréer automatiquement).
        try:
            vis = cv.getWindowProperty(win, cv.WND_PROP_VISIBLE)
        except cv.error:
            break
        if vis < 1:
            break

        k = cv.getTrackbarPos("Noyau", win)
        if k < 1:
            k = 1
            cv.setTrackbarPos("Noyau", win, k)
        if k % 2 == 0:
            k += 1
            if k > 11:
                k = 11
            cv.setTrackbarPos("Noyau", win, k)

        sigma_mult = max(1, cv.getTrackbarPos("SeuilSigma", win))
        iters = max(1, cv.getTrackbarPos("Iterations", win))

        # traiter
        eroded, mask, bg, stars, thr = process_image(image_u8, k, float(sigma_mult), iters)

        # construire affichage côte-à-côte: original | eroded | mask
        disp_orig = to_display(image_u8)
        disp_eroded = to_display(eroded)
        disp_mask = to_display(mask)

        # redimensionner pour affichage si trop large
        h, w = disp_orig.shape[:2]
        target_h = 600
        scale = target_h / max(h, 1)
        if scale < 1.0:
            new_w = int(w * scale)
            disp_orig = cv.resize(disp_orig, (new_w, target_h))
            disp_eroded = cv.resize(disp_eroded, (new_w, target_h))
            disp_mask = cv.resize(disp_mask, (new_w, target_h))

        combined = cv.hconcat([disp_orig, disp_eroded, disp_mask])

        # libellés au-dessus de chaque panneau: Original | Érodé | Masque
        w1 = disp_orig.shape[1]
        w2 = disp_eroded.shape[1]
        cv.putText(combined, "Original", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(combined, "Erode", (w1 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(combined, "Masque", (w1 + w2 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # annotation technique (paramètres)
        cv.putText(combined, f"Noyau={k} It={iters} Thresh={thr}", (10, combined.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv.imshow(win, combined)

        key = cv.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        if key == ord('s'):
            # sauvegarde des images actuelles
            base = os.path.splitext(os.path.basename(fits_path))[0]
            cv.imwrite(os.path.join(results_dir, f"{base}_eroded_k{k}_it{iters}_thr{int(thr)}.png"), eroded)
            cv.imwrite(os.path.join(results_dir, f"{base}_mask.png"), mask)
            cv.imwrite(os.path.join(results_dir, f"{base}_bg.png"), bg)
            print("Images sauvegardées dans:", results_dir)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()