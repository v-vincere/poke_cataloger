from ultralytics import YOLO

# Load a pretrained YOLO model (yolov8n.pt is a good starting point)
# Load a pretrained YOLOv11 model (yolov11n.pt is the smallest and fastest)
model = YOLO('yolo11n.pt')

# Train the model using our dataset
if __name__ == '__main__':
    results = model.train(
        data='/mnt/d/ml_poke/dataset.yaml',
        epochs=100,
        imgsz=640,
        project='/mnt/d/ml_poke/runs',
        name='card_detector_v1'
    )