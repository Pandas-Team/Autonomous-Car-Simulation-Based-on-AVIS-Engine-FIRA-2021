import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from time import sleep

def detect_edges(image):
    # detect edges
    image = cv2.Canny(image, 100, 250)
    return image

def region_of_interest(image):
    (height, width) = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[
        # (0, height),
        # (0, 170),
        # (65,115),
        # (256-65,115),
        # (width, 170),   
        # (width, height),
        (45, height),
        (80, 165),
        (256-80, 165),
        (256-45, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = image * (255-mask)
    return masked_image

def detect_lines(image):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    lines = cv2.HoughLinesP(image, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), [255,0,0], 3)

    return image, lines

def mean_lines(frame, lines):
    a = np.zeros_like(frame)
    try:
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
                if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                    continue
                if slope <= 0: # <-- If the slope is negative, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else: # <-- Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
        min_y = int(frame.shape[0] * (3 / 5)) # <-- Just below the horizon
        max_y = int(frame.shape[0]) # <-- The bottom of the image
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        cv2.line(a, (left_x_start, max_y), (left_x_end, int(min_y)), [255,255,0], 5)
        cv2.line(a, (right_x_start, max_y), (right_x_end, int(min_y)), [255,255,0], 5)
        c_p = (left_x_end+right_x_end)/2
    except:
        c_p = 128
        pass
    return a, c_p

def detect_yellow_line(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.blur(hsv_frame,(6,6))
    mask = cv2.inRange(hsv_frame, np.array([0,95,0]), np.array([31,255,255]))
    mask = cv2.GaussianBlur(mask , (7,7), 0)
    mask = cv2.medianBlur(mask , 7)
    result = cv2.bitwise_and(frame, frame, mask = mask)

    a = np.zeros_like(frame)
    try:
        frame = detect_edges(result)
        frame, lines = detect_lines(frame)

        left_line_x = []
        left_line_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
        min_y = int(frame.shape[0] * (1 / 5)) # <-- Just below the horizon
        max_y = int(frame.shape[0]) # <-- The bottom of the image
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        max_x = int(poly_left(max_y))
        min_x = int(poly_left(min_y))
        cv2.line(a, (max_x, max_y), (min_x, int(min_y)), [255,255,0], 5)
    except:
        pass
    try:
        slope = -(max_y-min_y)/(max_x-min_x)
    except:
        slope = 0
    return a,slope

def lane_change_to_left(car,yellow_slope):
    while yellow_slope > -0.1:
        
        car.setSteering(-50)

def lane_change_to_right(yellow_slope):
    while yellow_slope < 0.1:
        car.setSteering(+50)

def bypass_obstacle(position):
    if position == 'right':
        lane_change_to_left()
    else:
        lane_change_to_right()


def interval_avg(x,y):
    num_samples = 10
    if not x.size:
        return None
    step = (x.max() - x.min()) // num_samples
    x = np.sort(x)
    y = np.sort(y)
    intervals = np.arange(x.min(), x.max(), step)
    avgs_x = []
    avgs_y = []
    for i, inter in enumerate(intervals[:-1]):
        avg_y = np.mean(y[(x<intervals[i+1]) & (x>intervals[i])])
        avg_x = np.mean(x[(x<intervals[i+1]) & (x>intervals[i])])
        avgs_x.append(avg_x)
        avgs_y.append(avg_y)
    return [np.array(avgs_x), np.array(avgs_y)]

def find_position(white_where, yellow_where):
    if (white_where > 0) and (yellow_where > 0):
        pos = 1
    elif (white_where < 0) and (yellow_where > 0):
        pos = 2
    elif (white_where > 0) and (yellow_where < 0):
        pos = 3
    elif (white_where < 0) and (yellow_where < 0):
        pos = 4
    return pos

def car_status(actual_where, sensors):
    rounded_where = int(np.round(actual_where))
    if rounded_where == 3:
        if (sensors[1] == 1500) and (sensors[2] == 1500) :
            print("clear")
            return 1

        elif sensors[1] != 1500 or (sensors[2] != 1500):
            print('obstacle on right, turn left')
            return 2
    

    elif rounded_where == 2:
        if (sensors[0] == 1500) and (sensors[1] == 1500) :
            print('Clear, go back to right')

            return 3
        elif (sensors[1] != 1500):
            print('Obstacle Ahead')

            return 4


    elif rounded_where == 1:
        pass

    elif rounded_where == 4:
        pass

    
# def turn_left(counter):

    