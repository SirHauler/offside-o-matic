from typing import List
from ultralytics import YOLO

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

        self.model = YOLO("yolov8n.pt")


    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        
        result = self.model(input_image, size=640)
        
        return result.pandas().xyxy[0]

    

    


