import os

BASE_DIR = r"C:/Users/user/OneDrive/Desktop/Analateica"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Required model filenames
REQUIRED_MODELS = [
    "yolov8n-pose.onnx",
    "fire_small_yolo.onnx",
    "weapon_yolo.onnx",
    "cloth_yolo.onnx",
    "mediapipe_face_mesh_478.onnx",
    "movenet_har_fall_fight_run.onnx"
]

# 1) Create folder if missing
if not os.path.exists(MODEL_DIR):
    print(f"[CREATE] Creating folder: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
else:
    print(f"[OK] Folder exists: {MODEL_DIR}")

print("\nChecking required model files...\n")

# 2) Check file existence
missing = []
for model in REQUIRED_MODELS:
    path = os.path.join(MODEL_DIR, model)
    if os.path.exists(path):
        print(f"✔ FOUND: {model}")
    else:
        print(f"❌ MISSING: {model}")
        missing.append(model)

# 3) Summary
print("\n----------------------------------------")
if missing:
    print("❌ Some required models are missing:")
    for m in missing:
        print("   →", m)

    print("\nDownload links:")
    print("yolov8n-pose.onnx → https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.onnx")
    print("fire_small_yolo.onnx → https://github.com/ultralytics/assets/releases/download/v8.1.0/fire.onnx (rename)")
    print("weapon_yolo.onnx → https://github.com/ultralytics/assets/releases/download/v8.1.0/weapons.onnx (rename)")
    print("cloth_yolo.onnx → https://github.com/ultralytics/assets/releases/download/v8.1.0/clothes.onnx (rename)")
    print("mediapipe_face_mesh_478.onnx → https://github.com/tensorflow/tfjs-models/raw/master/face-landmarks-detection/mesh_model.onnx (rename)")
    print("movenet_har_fall_fight_run.onnx → https://huggingface.co/ibrahim-m/movenet-har-fall-fight-run/resolve/main/movenet_har_fall_fight_run.onnx")
else:
    print("🎉 ALL MODELS PRESENT — YOU ARE READY TO RUN THE SURVEILLANCE SYSTEM!")
print("----------------------------------------")
