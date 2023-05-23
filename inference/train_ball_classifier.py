from ultralytics import YOLO



def runYoloV8():
    model = YOLO("yolov8n.pt")    
    results = model.train(data="../datasets/soccerball/data.yaml", epochs=2, save=True, save_period=1, optimizer='AdamW')


def detectYoloV8(): 
    model = YOLO("runs/detect/train8/weights/best.pt")

def results():
    model = YOLO("runs/detect/train8/weights/best.pt")
    results = model.val()
    print(results)

# if __name__ == "__main__": 
#     # runYoloV8()
#     results()