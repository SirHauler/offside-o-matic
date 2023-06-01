import numpy as np
import norfair
import PIL
import sys
import cv2

from typing import List
from norfair import Detection
from PIL import Image


from soccer.ball import Ball 
from soccer.draw import Draw
from soccer.team import Team

TOP_LINE_Y = 500
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (1, 1, 1)

class OffsideLine:
    def __init__(self): 
        # self.detection = detection

        self.team = None
        self.last_man_left = None  # will be updated
        self.last_man_right = None


    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id: bool = False
    ) -> PIL.Image.Image:


        xy_left = self.last_man_left   # draw the line here 
        xy_right = self.last_man_right
        # TODO: correct for the vanishing point!

        cv2_frame = self.pil_to_cv2(frame)

        # calculate vanishing points and lines
        vanishing_points, lines = self.detect_vanishing_point(cv2_frame)

        pil_frame = self.cv2_to_pil(cv2_frame)

        draw = PIL.ImageDraw.Draw(pil_frame)  # testing for color conversion
        
        # coords_left = [xy_left, (0, 0)]
        # coords_right = [xy_right, (0, 0)]
        # draw.line(xy=coords_left, width=5)
        # draw.line(xy=coords_right, width=5)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y1 < TOP_LINE_Y and y2 < TOP_LINE_Y):
                continue
            points = [(x1, y1), (x2, y2)]
            draw.line(xy=points, fill=COLOR_WHITE, width=5)
            
        offside_line, got_line = self.get_line(lines=lines)
        if got_line: 
            draw.line(xy=offside_line, fill=COLOR_BLACK, width=5)
        
        vp_translated = (vanishing_points[0] + 10, vanishing_points[1] + 10)
        circle_points = [vanishing_points, vp_translated]
        draw.ellipse(xy=circle_points, fill=COLOR_WHITE, width=2)

        return pil_frame

    def get_last_man(
        self, players: List["Player"]
    ): 
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = -sys.maxsize, -sys.maxsize

        for player in players: 
            x1, y1 = player.detection.points[0]
            x2, y2 = player.detection.points[1]
            if x1 < min_x: 
                min_x = x1
                min_y = y1
            if x2 > max_x:
                max_x = x2
                max_y = y2

        # this is where the line will be drawn
        self.last_man_left = (min_x, min_y)  
        self.last_man_right = (max_x, max_y)
    

    @staticmethod
    def pil_to_cv2(pil_image):
        # Convert PIL image to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB image to BGR
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return bgr_image
    
    @staticmethod
    def cv2_to_pil(cv2_image):
        # Convert BGR image to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Create PIL Image from numpy array
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
            
    @staticmethod
    def detect_vanishing_point(cv2_image):

        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1=90, threshold2=200, apertureSize=5)
        
        # Apply HoughLinesP (probabilistic) to detect lines
        # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200, minLineLength=200, maxLineGap=100)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=600, maxLineGap=30)
        
        # Find the vanishing point by averaging the intersection of lines
        num_lines = len(lines)
        x_sum = 0
        y_sum = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # eliminating lines from the top
            if (y1 < TOP_LINE_Y and y2 < TOP_LINE_Y): 
                continue
            x_sum += (x1 + x2)
            y_sum += (y1 + y2)
        
        vanishing_x = int(x_sum / (2 * num_lines))
        vanishing_y = int(y_sum / (2 * num_lines))

        vanishing_points = (vanishing_x, vanishing_y)

        return vanishing_points, lines
    
    @staticmethod
    def get_line(
        lines
    ): 
        got_line = False 
        min_x2, min_y1 = sys.maxsize, sys.maxsize
        min_x1, min_y2 = sys.maxsize, sys.maxsize
        for line in lines: 
            x1, y1, x2, y2 = line[0]
            if (y1 < TOP_LINE_Y and y2 < TOP_LINE_Y):
                continue
            if y1 < min_y1 or x2 < min_x2:
            # if x1 < min_x1 or x2 < min_x2:
                min_x1 = x1
                min_y1 = y1
                min_x2 = x2
                min_y2 = y2

        if min_y1 != sys.maxsize or min_x2 != sys.maxsize or min_x1 != sys.maxsize or min_y2 != sys.maxsize:
            got_line = True
        
        offside_line = [(min_x1, min_y1), (min_x2, min_y2)]
        return offside_line, got_line