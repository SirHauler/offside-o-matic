from ultralytics import YOLO



def runYoloV8():
    model = YOLO("yolov8n.pt")    

    results = model.train(data="../datasets/soccerball/data.yaml", epochs=10)




if __name__ == "__main__": 
    runYoloV8()