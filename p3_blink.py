import os
import time
import cv2 as cv

# -----------------------------
# GUI: choisir un dossier (Tkinter)
# -----------------------------
def choose_folder_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as e:
        raise ImportError(
            "Tkinter n'est pas installé. Sur Ubuntu: sudo apt install python3-tk"
        ) from e

    root = tk.Tk()
    root.title("Phase 3 - Choisir le dossier de résultats")
    root.geometry("520x140")
    root.resizable(False, False)

    folder_var = tk.StringVar(value="")

    def browse():
        path = filedialog.askdirectory(title="Sélectionne le dossier contenant original.png et final_star_reduced.png")
        if path:
            folder_var.set(path)

    def validate():
        path = folder_var.get().strip()
        if not path:
            messagebox.showerror("Erreur", "Aucun dossier sélectionné.")
            return

        orig = os.path.join(path, "original.png")
        final = os.path.join(path, "final_star_reduced.png")

        if not os.path.isfile(orig):
            messagebox.showerror("Erreur", f"original.png introuvable dans:\n{path}")
            return
        if not os.path.isfile(final):
            messagebox.showerror("Erreur", f"final_star_reduced.png introuvable dans:\n{path}")
            return

        root.selected_folder = path
        root.destroy()

    # Widgets
    tk.Label(root, text="Dossier de résultats (doit contenir original.png et final_star_reduced.png):").pack(pady=(12, 6))
    entry = tk.Entry(root, textvariable=folder_var, width=70)
    entry.pack(pady=(0, 8))

    btn_frame = tk.Frame(root)
    btn_frame.pack()

    tk.Button(btn_frame, text="Parcourir…", command=browse, width=14).pack(side=tk.LEFT, padx=6)
    tk.Button(btn_frame, text="Valider", command=validate, width=14).pack(side=tk.LEFT, padx=6)

    root.selected_folder = None
    root.mainloop()

    if not getattr(root, "selected_folder", None):
        return ""
    return root.selected_folder


# -----------------------------
# Blink viewer (OpenCV)
# -----------------------------
def run_blink_viewer(folder: str, blink_ms: int = 1100) -> None:
    orig_path = os.path.join(folder, "original.png")
    final_path = os.path.join(folder, "final_star_reduced.png")

    orig = cv.imread(orig_path, cv.IMREAD_COLOR)
    final = cv.imread(final_path, cv.IMREAD_COLOR)

    if orig is None:
        raise FileNotFoundError(f"Impossible de lire {orig_path}")
    if final is None:
        raise FileNotFoundError(f"Impossible de lire {final_path}")

    # Sécurité: même taille
    if orig.shape[:2] != final.shape[:2]:
        final = cv.resize(final, (orig.shape[1], orig.shape[0]), interpolation=cv.INTER_AREA)

    window_name = "Blink (space=pause, o=orig, f=final, +/- speed, q=quit)"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    show_final = False
    paused = False
    last_switch = time.time()

    while True:
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        now = time.time()
        if not paused and (now - last_switch) * 1000.0 >= blink_ms:
            show_final = not show_final
            last_switch = now

        img = final if show_final else orig
        label = "FINAL (reduced)" if show_final else "ORIGINAL"

        frame = img.copy()
        cv.putText(
            frame, label, (20, 40),
            cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA
        )
        cv.putText(
            frame, f"Blink: {blink_ms} ms", (20, 80),
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA
        )

        cv.imshow(window_name, frame)

        key = cv.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('o'):
            show_final = False
            paused = True
        elif key == ord('f'):
            show_final = True
            paused = True
        elif key == ord('+') or key == ord('='):
            blink_ms = max(200, blink_ms - 200)
        elif key == ord('-'):
            blink_ms = min(5000, blink_ms + 200)

    cv.destroyAllWindows()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    folder = choose_folder_dialog()
    if not folder:
        print("❌ Aucun dossier sélectionné. Fin du programme.")
    else:
        print(f"📁 Dossier sélectionné : {folder}")
        run_blink_viewer(folder, blink_ms=1100)
