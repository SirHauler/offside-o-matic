from typing import List
from ultralytics import YOLO
import ultralytics

import numpy as np
import pandas as pd
import torch

from inference.base_detector import BaseDetector


class YoloV8(BaseDetector):
    def __init__(
        self,
    ): 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # self.model = YOLO('yolov8n.pt')
        # self.model = YOLO('yolov8s.pt')
        self.model = YOLO('football-player.pt')

        self.second_model = torch.hub.load(
            "ultralytics/yolov5", "yolov5x", pretrained=True
        )

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        # result = self.model(input_image, size=640)

        results = self.model(input_image)

        # second_results = self.second_model(input_image, size=640)
        # Create a pandas DataFrame from the dictionary
        # df = pd.DataFrame(data)
        # return result.pandas().xyxy[0]
        store = {'xmin' : [], 
                 'ymin' : [], 
                 'xmax' : [], 
                 'ymax' : [], 
                 'confidence' : [], 
                 'class' : [], 
                 'name' : []}
                
        for result in results:
            boxes = result.boxes
            # print(boxes)
            N, _ = boxes.shape
            # print("Shape", boxes.xyxy.shape)

            # for i in range(N):
            # # if (int(boxes.cls[0]) == 0): 
            #     store['name'].append("sports ball")
            #     store['xmin'].append(float(boxes.xyxy[0][0]))
            #     store['ymin'].append(float(boxes.xyxy[0][1]))
            #     store['xmax'].append(float(boxes.xyxy[0][2]))
            #     store['ymax'].append(float(boxes.xyxy[0][3]))
            #     store['confidence'].append(float(boxes.conf[0]))
            #     store['class'].append(int(boxes.cls[0]))

            for i in range(N):
            # if (int(boxes.cls[0]) == 0): 
                store['name'].append("sports ball")
                store['xmin'].append(float(boxes.xyxy[i][0]))
                store['ymin'].append(float(boxes.xyxy[i][1]))
                store['xmax'].append(float(boxes.xyxy[i][2]))
                store['ymax'].append(float(boxes.xyxy[i][3]))
                store['confidence'].append(float(boxes.conf[i]))
                store['class'].append(int(boxes.cls[i]))
                        
                        
                # elif (int(boxes.cls[i] == 32)): 
                #     store['name'].append("N/A")
                # else: 
                #     store['name'].append("N/A")

        print(store)

        # print(pd.DataFrame(store))
        # print(second_results.pandas().xyxy[0])

        return pd.DataFrame(store)
    

    


