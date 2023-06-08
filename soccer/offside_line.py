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

TOP_LINE_Y = 700
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (1, 1, 1)

class OffsideLine:
    def __init__(self): 
        # self.detection = detection

        self.team = None
        self.last_man_left = None  # will be updated
        self.last_man_right = None
        self.players = None# list of players


    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id: bool = False
    ) -> PIL.Image.Image:


        # xy_left = self.last_man_left   # draw the line here 
        # xy_right = self.last_man_right
        # TODO: correct for the vanishing point!

        cv2_frame = self.pil_to_cv2(frame)

        # calculate vanishing points and lines
        lines = self.detect_vanishing_point_lines(cv2_frame)

        pil_frame = self.cv2_to_pil(cv2_frame)

        draw = PIL.ImageDraw.Draw(pil_frame)  # testing for color conversion
        
        for line in lines:
            x1, y1, x2, y2 = line

            if (y1 < TOP_LINE_Y and y2 < TOP_LINE_Y):
                continue
            points = [(x1, y1), (x2, y2)]
            draw.line(xy=points, fill=COLOR_WHITE, width=5)
            
        offside_line, got_line = self.get_line(lines=lines)

        # if got_line: 
        draw.line(xy=offside_line, fill=COLOR_BLACK, width=5)
        
        # vp_translated = (vanishing_points[0] + 10, vanishing_points[1] + 10)
        # circle_points = [vanishing_points, vp_translated]
        # draw.ellipse(xy=circle_points, fill=COLOR_WHITE, width=2)

        # # find last man
        # last_man, foot_point = self.get_last_man(offside_line) # returns point
        # print("Foot_point(): ", foot_point)
        # if foot_point is not None:
        #     foot_points = [foot_point, (foot_point[0] + 8, foot_point[1] + 8)]
        #     draw.ellipse(xy=foot_points, fill=COLOR_BLACK, width=2)
        
        return pil_frame

    def get_last_man(
        self, 
        offside_line
    ): 
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = -sys.maxsize, -sys.maxsize

        last_man_left, point = self.find_closest_player(offside_line)

        return last_man_left, point

    def save_players(self, players):
        self.players = players

    def find_closest_player(self, line):

        min_distance = float('inf')
        closest_point = None
        closest_player = None

        for player in self.players:
            # look at right and left foot
            x_min, _ = player.detection.points[0]
            x_max, y_max = player.detection.points[1]

            
            # skip any lines that have out of bound coordinates
            if not self.in_bounds(line[0]) or not self.in_bounds(line[1]):
                continue

            right_foot_x, right_foot_y = x_max, y_max # bottom right corner of bounding box
            left_foot_x, left_foot_y = x_min, y_max  # bottom left corner of boudning box

            print("Left Foot: ", left_foot_x, left_foot_y)
            print("Right Foot: ", right_foot_x, right_foot_y)
            
            left = self.distance(line, (left_foot_x, left_foot_y))
            right = self.distance(line, (right_foot_x, right_foot_y))


            if left < min_distance:
                min_distance = left
                closest_point = (left_foot_x, left_foot_y)
                closest_player = player

            if right < min_distance:
                min_distance = right
                closest_point = (right_foot_x, right_foot_y)
                closest_player = player

        return player, closest_point



    @staticmethod 
    def in_bounds(point) -> bool:
        """Checks if point is in bounds

        Args:
            point (tuple): x, y coords

        Returns:
            bool: is in bounds
        """
        x, y = point
        return x < 200000 and y < 200000


    @staticmethod
    def distance(line, point): 
        x1, y1 = line[0][0], line[0][1]
        x2, y2 = line[1][0], line[1][1]
        x, y = point
        print(x, y, x1, y1, x2, y2)
        distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        
        return distance


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
    def detect_vanishing_point_lines_normal(cv2_image):

        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/360, threshold=300)

        formated_lines = []
        for line in lines:
            rho, theta = line[0][0], line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            euclid_line = [x1, y1, x2, y2]
            formated_lines.append(euclid_line)
        # Draw the lines on the original image
        return formated_lines
    
            

    @staticmethod
    def get_line(
        lines
    ): 
        got_line = False 
        min_x2, min_y1 = sys.maxsize, sys.maxsize
        min_x1, min_y2 = sys.maxsize, sys.maxsize
        for line in lines: 
            x1, y1, x2, y2 = line
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
    
    @staticmethod
    def detect_vanishing_point_lines(cv2_image):

        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1=90, threshold2=200, apertureSize=5)
        
        # Apply HoughLinesP (probabilistic) to detect lines
        # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200, minLineLength=200, maxLineGap=100)

        lines = cv2.HoughLinesP(edges, rho=1.25, theta=np.pi/270, threshold=350, minLineLength=700, maxLineGap=27)
        
        # Find the vanishing point by averaging the intersection of lines

        num_lines = len(lines)
        x_sum = 0
        y_sum = 0
        formated_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # eliminating lines from the top
            if (y1 < TOP_LINE_Y and y2 < TOP_LINE_Y): 
                continue
            x_sum += (x1 + x2)
            y_sum += (y1 + y2)
            euclid_line = [x1, y1, x2, y2]
            formated_lines.append(euclid_line)
        
        vanishing_x = int(x_sum / (2 * num_lines))
        vanishing_y = int(y_sum / (2 * num_lines))

        vanishing_points = (vanishing_x, vanishing_y)

        return formated_lines
    