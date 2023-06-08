import argparse

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from PIL import Image

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5, YoloV8
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)

from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass
from soccer.offside_line import OffsideLine

'''
Adaptation of run_video.py file for singular image instead
of video!
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

image = cv2.imread(args.video, cv2.IMREAD_COLOR)

fps = 60

# Object Detectors
player_detector = YoloV5()
# ball_detector = YoloV8() # TODO: update the model to be the trained version on the ROBOFLOW dataset
ball_detector = YoloV5()

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
chelsea = Team(
    name="Chelsea",
    abbreviation="CHE",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(name="Man City", abbreviation="MNC", color=(240, 230, 188))
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city


# Offside Line
offside_line = OffsideLine()

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=20,
    hit_counter_max=2000,
)


motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()


    # assert i != 20

    # Get Detections
players_detections = get_player_detections(player_detector, image)
ball_detections = get_ball_detections(ball_detector, image)
detections = ball_detections + players_detections

# Update trackers
    

# Match update
ball = get_main_ball(ball_detections)
players = Player.from_detections(detections=players_detections, teams=teams)
match.update(players, ball)

# Draw
image = PIL.Image.fromarray(image)


#TODO: Draw Offside Line for testing
offside_line.save_players(players)
image = offside_line.draw(image)


if args.possession:
    image = Player.draw_players(
        players=players, frame=image, confidence=False, id=True
    )

    image = path.draw(
        img=image,
        detection=ball.detection,
        coord_transformations=coord_transformations,
        color=match.team_possession.color,
    )

    if ball:
        image = ball.draw(image)

if args.passes:
    pass_list = match.passes

    image = Pass.draw_pass_list(
        img=image, passes=pass_list, coord_transformations=coord_transformations
    )

    image = match.draw_passes_counter(
        image, counter_background=passes_background, debug=False
    )

image = np.array(image)

# Write video
cv2.imwrite('result.jpg', image)