from ultralytics import YOLO

# 1. 加载模型
model_path = "yolo11n-fce.yaml"
model = YOLO(model_path)

from ultralytics import YOLO

# 1. 加载模型
model_path = "yolo11n-fce.yaml"
model = YOLO(model_path)

model.info(detailed=True, verbose=True)