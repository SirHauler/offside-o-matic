from typing import List

import numpy as np
import pandas as pd
import torch
import yolov5

from inference.base_detector import BaseDetector


class YoloV5(BaseDetector):
    def __init__(
        self,
        model_path: str = None,
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will download the model with COCO weights
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        
        self.custom = False

        if model_path:
            # doesn't work atm
            # self.model = torch.hub.load("ultralytics/yolov5", "custom", path='ball.pt', force_reload=True)
            model = yolov5.load('keremberke/yolov5n-football')

            model.conf = 0.25  # NMS confidence threshold
            model.iou = 0.45  # NMS IoU threshold
            model.agnostic = False  # NMS class-agnostic
            model.multi_label = False  # NMS multiple labels per box
            model.max_det = 1000  # maximum number of detections per image
            self.model = model
            self.custom = True

        else:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "yolov5x", pretrained=True, force_reload=True
            )

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        result = self.model(input_image)

        if self.custom: 
            df = result.pandas().xyxy[0] # first img
            classes_to_exclude = ['player']
            df_filtered = df[~df['name'].isin(classes_to_exclude)]
            # print(df_filtered)
            return df_filtered
        
        return result.pandas().xyxy[0]
