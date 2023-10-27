from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("weights/best_v3.pt")
    # model = YOLO("weights/best_v1.pt")
    results = model.train(data="datasets/barCode.yaml", epochs=300, batch=-1, optimizer="Adam", lr0=1e-4)

    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered

