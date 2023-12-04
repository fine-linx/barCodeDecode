from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO("weights/best_v6.pt")
    # model = YOLO("weights/yolov8s.pt")
    model = YOLO("runs/detect/train6/weights/best.pt")
    results = model.train(data="datasets/barCode.yaml",
                          epochs=10,
                          batch=-1,
                          # cache=True,
                          patience=6,
                          optimizer="Adam",
                          lr0=1e-4
                          )
    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
