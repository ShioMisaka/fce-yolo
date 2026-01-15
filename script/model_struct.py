from ultralytics.nn.tasks import DetectionModel

# 直接创建模型，verbose=True 会打印每一层的信息
model = DetectionModel('yolo11n-fce.yaml', ch=3, nc=None, verbose=True)