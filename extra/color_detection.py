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
        (0, height),
        (0, height * 7/10),
        (width * 1 / 4, height * 1 / 2),
        (width * 3 / 4, height * 1 / 2),
        (width, height * 7/10),   
        (width, height),
        (width-45, height),
        (width-84, 165),
        (84, 165),
        (45, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = image * mask
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
    except:
        pass
    return a

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

for file in os.listdir('./datas/Race Track1(No obstacles)'):
    frame = cv2.imread('./datas/Race Track1(No obstacles)/'+file)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.blur(hsv_frame,(6,6))
    mask = cv2.inRange(hsv_frame, np.array([0,33,115]), np.array([180,70,206]))
    mask = cv2.GaussianBlur(mask , (7,7), 0)
    mask = cv2.medianBlur(mask , 7)
    result = cv2.bitwise_and(frame, frame, mask = mask)
    result[0:100,:]=0
    cv2.imshow('result',result)

    frame_edge = detect_edges(result)
    roi = region_of_interest(frame_edge)
    cv2.imshow('roi',roi*255)

    frame_edge, lines = detect_lines(roi)
    final_frame = mean_lines(frame_edge, lines)

    a,slope = detect_yellow_line(frame)
    print(slope)
    cv2.imshow('frame',frame)
    cv2.imshow('frame two lines',final_frame)
    if slope<0:
        position = 'left'
    else:
        position = 'right'
    cv2.putText(a, position, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('yellow line', a)
    cv2.waitKey()

