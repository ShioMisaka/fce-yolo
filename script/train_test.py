from ultralytics import YOLO

# 1. 加载模型
model_path = "yolo11n-fce.yaml"
model = YOLO(model_path)

model.train(
    data="/home/shiomisaka/workplace/ai-playground/datasets/MY_TEST_DATA/data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu"
)