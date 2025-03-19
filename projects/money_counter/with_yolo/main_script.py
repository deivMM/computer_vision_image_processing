###
### Utilizar cvat para etiquetar las imagenes !!!!
###

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

# Train the model on your custom dataset

model.train(
    data='dataset/dataset.yaml',  # Define dataset in a .yaml file
    epochs=25                        # Number of training epochs
)

# model.train(
#     data='dataset/dataset.yaml',  # Define dataset in a .yaml file
#     epochs=20,                         # Number of training epochs
#     batch=16,                          # Batch size
#     imgsz=640,                         # Image size
#     device="cuda"                       # Use GPU if available, otherwise "cpu"
# )
