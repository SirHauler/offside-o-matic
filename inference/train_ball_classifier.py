from ultralytics import YOLO

def runYoloV8():
    model = YOLO("yolov5x.pt")    
    results = model.train(data="/home/ubuntu/cs231n/offside-o-matic/datasets/soccerball/yolov8/data.yaml", epochs=10, save=True, save_period=1, optimizer='AdamW')

def detectYoloV8(): 
    model = YOLO("runs/detect/train8/weights/best.pt")

def results():
    model = YOLO("runs/detect/train8/weights/best.pt")
    results = model.val()
    print(results)

if __name__ == "__main__": 
    runYoloV8()