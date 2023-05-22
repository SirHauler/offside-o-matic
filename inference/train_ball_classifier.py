from ultralytics import YOLO



def runYoloV8():
    model = YOLO("yolov8n.pt")    
    results = model.train(data="../datasets/soccerball/data.yaml", epochs=2, save=True, save_period=1, optimizer='AdamW')


if __name__ == "__main__": 
    runYoloV8()