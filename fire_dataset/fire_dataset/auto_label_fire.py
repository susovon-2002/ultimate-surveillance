import cv2
import os
import numpy as np

# FOLDERS (modify only if needed)
ROOT = r"C:\Users\user\OneDrive\Desktop\Analateica\fire_dataset\fire_dataset"
IMG_DIR = os.path.join(ROOT, "images", "train")
LBL_DIR = os.path.join(ROOT, "labels", "train")

os.makedirs(LBL_DIR, exist_ok=True)

def find_fire_boxes(img):
    """Returns YOLO bounding boxes detected using fire color threshold."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # FIRE COLOR RANGE (tuned for bright yellow/orange/red flames)
    lower = np.array([5, 80, 80])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Find fire contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    boxes = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:  # ignore small noise
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        # Convert to YOLO format (normalized)
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        boxes.append([xc, yc, nw, nh])

    return boxes

def auto_label():
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg"))]

    print(f"🔥 Found {len(images)} images in train folder")

    for img_name in images:
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LBL_DIR, img_name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read {img_name}")
            continue

        boxes = find_fire_boxes(img)

        with open(label_path, "w") as f:
            for b in boxes:
                f.write(f"0 {b[0]} {b[1]} {b[2]} {b[3]}\n")

        if boxes:
            print(f"✔ Labeled fire in {img_name} ({len(boxes)} boxes)")
        else:
            print(f"⚠ No fire found in {img_name} (empty label)")

    print("\n✅ Auto-label complete!\nYOLO labels saved in:", LBL_DIR)


if __name__ == "__main__":
    auto_label()
