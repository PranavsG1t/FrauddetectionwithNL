import random 
from pathlib import Path
import cv2
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("rawdata/midv2020")
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "document_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GENUINE_DIR = OUT_DIR / "genuine"
TAMPER_DIR = OUT_DIR / "tampered"

RANDOM_STATE = 42
IMG_EXTENSION = {".jpg", ".jpeg", ".png"}
TAMPERS_PER_IMAGE = 2
PATCH_SIZE_FRAC = (0.15, 0.30)

def find_genuine_inages(raw_dir: Path) -> list[Path]:
    """Find all genuine images in the raw data directory."""
    images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSION]
    if not images:
        raise FileNotFoundError(f"No images found in {raw_dir}. Please check the path and ensure that the images are present.")
    return images

def copy_move_tamper(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Copy a random rectangular patch on random location in the image."""
    h, w = img.shape[:2]
    short_side = min(h, w)
    patch_frac = rng.uniform(*PATCH_SIZE_FRAC)
    ph = pw = max(8, int(short_side * patch_frac))
    src_y =rng.randint(0, h- ph)
    src_x = rng.randint(0, w- pw)
    patch = img[src_y:src_y+ph, src_x:src_x+pw].copy()

    #keep destination reasonably far from the source so the copy is visible
    dist_y, dist_x = src_y, src_x
    for _ in range(10):
        dist_y = rng.randint(0, h- ph)
        dist_x = rng.randint(0, w- pw)
        if abs(dist_y - src_y) > ph or abs(dist_x - src_x)>pw:
            break
    tampered = img.copy()
    tampered[dist_y:dist_y+ph, dist_x:dist_x+pw] = patch
    
    blurred_patch = cv2.GaussianBlur(tampered[dist_y:dist_y+ph, dist_x:dist_x+pw], (9, 9), sigmaX=2.5)
    tampered[dist_y:dist_y+ph, dist_x:dist_x+pw] = blurred_patch
    return tampered

def main():
    GENUINE_DIR.mkdir(parents=True, exist_ok=True)
    TAMPER_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_STATE)
    print(f"Finding genuine images in {RAW_DIR}...")
    images = find_genuine_inages(RAW_DIR)
    print(f"Found {len(images)} genuine images.")

    rows =[]
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skiping unreadable file: {img_path}")
            continue

        genuine_name = f"{i:05d}.jpg"
        genuine_out = GENUINE_DIR/genuine_name
        cv2.imwrite(str(genuine_out), img)
        rows.append({"path": str(genuine_out), "label": 0, "source_id": i})

        for t in range(TAMPERS_PER_IMAGE):
            tampered_img = copy_move_tamper(img, rng)
            tampered_name = f"{i:05d}_tampered_{t}.jpg"
            tampered_out = TAMPER_DIR/tampered_name
            cv2.imwrite(str(tampered_out), tampered_img)
            rows.append({"path": str(tampered_out), "label": 1, "source_id": i})

        if (i+1) %100 == 0:
            print(f"Processed {i+1}/{len(images)} images.")
    manifest = pd.DataFrame(rows) #making a dataframe which contains the path and label of each image 0 for genuine and 1 for tampered
    print(f"\nTotal images: {len(manifest)}"
          f"(genuine: {(manifest.label==0).sum()}, tampered: {(manifest.label==1).sum()})")
    
    src_ids = manifest["source_id"].unique()
    train_ids, temp_ids = train_test_split(src_ids, test_size=0.3, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=RANDOM_STATE)
    manifest["split"] = np.where(manifest["source_id"].isin(train_ids), "train",
                        np.where(manifest["source_id"].isin(val_ids), "val", "test"))
    full_manifest = manifest
    full_manifest.to_csv(OUT_DIR/"manifest.csv", index=False)
    print(f"\nSaved manifest to {OUT_DIR/'manifest.csv'}")
    print(full_manifest.groupby(["split", "label"]).size())

if __name__ == "__main__":
    main()