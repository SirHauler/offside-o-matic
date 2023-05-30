import numpy as np
import norfair
import PIL
import sys


from typing import List
from norfair import Detection


from soccer.ball import Ball 
from soccer.draw import Draw
from soccer.team import Team


class OffsideLine:
    def __init__(self): 
        # self.detection = detection

        self.team = None
        self.last_man = None  # will be updated


    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id: bool = False
    ) -> PIL.Image.Image:

        xy = self.last_man   # draw the line here 

        # TODO: correct for the vanishing point!

        H, W = 100, 20
        color = (255, 255, 255)

        return Draw.draw_rectangle(
            img=frame, 
            origin=xy, 
            width=W, 
            height=H, 
            color=color
        )

    def get_last_man(
        self, players: List["Player"]
    ): 
        mx, my = sys.maxsize, sys.maxsize
        for player in players: 
            x1, y1 = player.detection.points[0]
            if x1 < mx: 
                mx = x1
                my = y1
        print("Last Man Left: ", (mx, my))
        # this is where the line will be drawn
        self.last_man = (mx, my)  
            

        
        
        
        # return tuple of (xmin, ymin, xmax, ymax)
        