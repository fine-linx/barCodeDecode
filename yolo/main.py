from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("weights/yolov8n.pt")
    # model = YOLO("weights/best_v1.pt")
    results = model.train(data="datasets/barCode.yaml", epochs=300, batch=-1, optimizer="Adam", lr0=0.001)

    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered

