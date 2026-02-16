from ultralytics import YOLO
import os

def run_training():

    DATA_YAML = r"C:/Users/centr/Downloads/tauhid/data.yaml"
    SAVE_DIR = r"C:/Users/centr/Downloads/tauhid/train_results"
    MODEL_NAME = "elephant_yolov8n_multi_class"

    print("\n🔍 Checking dataset directories...\n")

    base = "C:/Users/centr/Downloads/tauhid"
    required = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels"
    ]

    for folder in required:
        path = os.path.join(base, folder)
        print(("✔" if os.path.exists(path) else "❌") + " " + path)

    print("\n🚀 Starting YOLOv8 Training...\n")

    model = YOLO("yolov8n.pt")  # Pretrained weights

    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=80,
        batch=16,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,
        device=0,
        project=SAVE_DIR,
        name=MODEL_NAME,

        fliplr=0.5,
        translate=0.15,
        scale=0.3,
        mosaic=1.0,
        close_mosaic=10,

        cache=True,
        workers=0,          # 🔥 FIXED multiprocessing crash
        deterministic=True,
    )

    print("\n🎉 TRAINING SUCCESSFULLY COMPLETED!")
    print(f"📁 Output saved → {SAVE_DIR}/{MODEL_NAME}")


if __name__ == "__main__":  # 🔥 Required for Windows
    run_training()
