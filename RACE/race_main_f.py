import AVISEngine
import math
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
np.seterr(all="ignore")

###############################################
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
    num_samples = 5
    step = (x.max() - x.min()) // num_samples
#     intervals = np.
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
###############################################


#Calling the class
car = AVISEngine.car()
mean_obstacle = 0
yellow_roi = 0
LOGGING = False
#connecting to the server (Simulator)
car.connect("127.0.0.1", 25001)


if LOGGING : 
    columns = ['POSITION','SECOND_PXL_list','CURRENT_PXL','STEER']
    logged_data = pd.DataFrame(columns = columns)

REFRENCE = 128
CURRENT_PXL = 128
SECOND_PXL = 128

CURRENT_PXL_list = []
SECOND_PXL_list = []
steer_list = []
position_list = []

#Counter variable
counter = 0
slope = 1
debug_mode = True
#control part
kp = 2.2
ki = 0.2
kd = 0.2
previous_error = 0
integral = 0
steer = 0
dt = 0.05
sensors = [1500,1500,1500]
sensors_array = np.array([1500,1500,1500])
steer_array = np.array(0)
where_avg = 0.3
position = 'right'
#sleep for 3 second to make sure that client connected to the simulator 
time.sleep(3)
time1 = time.time()
try:
    while(True):  
        car.setSpeed(90)
        car.setSteering(0)
        car.getData()
        if(counter > 4):
            # getting data
            car.getData()
            sensors = car.getSensors() 
            sensors_array = np.round(where_avg * (np.array(sensors)) + (1-where_avg) * sensors_array, 1)
            frame = car.getImage()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # hsv_frame = cv2.medianBlur(hsv_frame, 7)

            mask = cv2.inRange(hsv_frame, np.array([100,10,25]), np.array([120,50,60]))
            kernel = np.ones((2,2),np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            kernel = np.ones((3,3),np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            # mask = cv2.medianBlur(mask, 13)
            lane_contours, _ = cv2.findContours(mask[130:200,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_lanes = sorted(lane_contours, key=cv2.contourArea, reverse=True)

            if len(sorted_lanes) > 1:
                right_lane_mask = cv2.drawContours(np.zeros((70,256)), sorted_lanes, 0, 255, -1)
                left_lane_mask = cv2.drawContours(np.zeros((70,256)), sorted_lanes, 1, 255, -1)

            elif len(sorted_lanes) == 1:
                right_lane_mask = cv2.drawContours(np.zeros((70,256)), sorted_lanes, 0, 255, -1)


            CURRENT_PXL = np.mean(np.where(right_lane_mask>0), axis=1)[1]
            SECOND_PXL = np.mean(np.where(left_lane_mask>0), axis=1)[1]
            CURRENT_PXL = np.nan_to_num(CURRENT_PXL, nan = 128)
            SECOND_PXL = np.nan_to_num(SECOND_PXL, nan = 128)

            yellow_mask = cv2.inRange(hsv_frame, np.array([28,115,154]), np.array([31,180,255]))
            YELLOW_PXL = np.mean(np.where(yellow_mask[140:190,:]>0), axis=1)[1]
            YELLOW_PXL = np.nan_to_num(YELLOW_PXL, nan = 128)
            if YELLOW_PXL>128:
                position = 'left'
            else:
                position = 'right'
            
            obs_yellow = np.mean(np.where(yellow_mask[65:170,:]>0), axis=1)[1]
            obs_yellow = np.nan_to_num(obs_yellow, nan = 128)

            # obstacle
            obstacle_mask = cv2.inRange(hsv_frame, np.array([95,0,95]), np.array([180,20,160]))
            kernel = np.ones((2,2),np.uint8)
            obstacle_mask = cv2.erode(obstacle_mask, kernel, iterations=1)
            kernel = np.ones((3,3),np.uint8)
            obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
            points, _ = cv2.findContours(obstacle_mask[50:200,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_points = sorted(points, key=len)
            try:
                if cv2.contourArea(sorted_points[-1])>10:
                    x,y,w,h = cv2.boundingRect(sorted_points[-1])
                    mean_obstacle = x + w//2
            except:
                pass
            
            if (position=='left'):
                if ((sensors_array[0]<1450 or sensors_array[1]<1450) and (mean_obstacle<obs_yellow)):
                    error = (REFRENCE - SECOND_PXL)
                    time1 = time.time()
                    steer = -(kp * error)
                    steer_array = np.round(0.85 * (np.array(steer)) + 0.15 * steer_array, 1)
                    car.setSteering(int(steer_array))

                elif((time.time()-time1)>0.8 and (min(sensors_array) > 1450)):
                    error = (REFRENCE - SECOND_PXL)
                    steer = -(kp * error)
                    steer_array = np.round(0.85 * (np.array(steer)) + 0.15 * steer_array, 1)
                    car.setSteering(int(steer_array))
                
                else:
                    error = REFRENCE - CURRENT_PXL 
                    steer = -(kp * error)
                    steer_array = np.round(0.85 * (np.array(steer)) + 0.15 * steer_array, 1)
                    car.setSteering(int(steer_array))
                
            else:
                if ((sensors_array[2]<1450 or sensors_array[1]<1450) and (mean_obstacle>obs_yellow)):
                    error = (REFRENCE - SECOND_PXL)
                    time1 = time.time()
                    steer = -(kp * error)
                    steer_array = np.round(0.85 * (np.array(steer)) + 0.15 * steer_array, 1)
                    car.setSteering(int(steer_array))
                
                else:
                    error = REFRENCE - CURRENT_PXL 
                    steer = -(kp * error)
                    steer_array = np.round(0.85 * (np.array(steer)) + 0.15 * steer_array, 1)
                    car.setSteering(int(steer_array))


            car.setSpeed(90)

            # display
            cv2.putText(frame, position, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame,(0,50),(256,100),(0,255,255),1)
            frame = cv2.rectangle(frame,(0,150),(256,200),(0,255,255),1)
            # ImShow
            show_img = np.concatenate((frame, np.dstack([mask,mask,mask])), axis=1)
            h1_axis = np.concatenate((left_lane_mask, right_lane_mask), axis=1)
            h2_axis = np.concatenate((obstacle_mask, yellow_mask), axis=1)
            show_mask = np.concatenate((h1_axis, h2_axis), axis=0) 

            cv2.imshow('img',show_img)
            cv2.imshow('info',show_mask)
            key = cv2.waitKey(1)
            if key == ord('w'):
                np.save('masks.npy', show_mask)
                cv2.imwrite('./race_frame.png', frame)
            
        

            os.system('cls')
            print(f'Current : {CURRENT_PXL}')
            print(f'Second : {SECOND_PXL}')
            print(f'Error : {error}')
            print(f'Steer : {steer}')
            print(f'Yellow line : {YELLOW_PXL}')
            print(f'Obstacle yellow : {obs_yellow}')
            print(f'Mean obstacle : {mean_obstacle}')
            print(f'Position : {position}')
            print(sensors_array)

        counter = counter + 1

        # SECOND_PXL_list.append(SECOND_PXL)
        # CURRENT_PXL_list.append(CURRENT_PXL)
        # steer_list.append(steer)
        # position_list.append(position)
        if LOGGING : 
            data = [position , SECOND_PXL, CURRENT_PXL, steer]
            new_row = pd.Series(index = columns, data = data)
            logged_data = logged_data.append(new_row,ignore_index=True)
            logged_data.to_excel('Output.xlsx')

        
        
finally:

    car.stop()
    
    # position_list = np.array(position_list)
    # SECOND_PXL_list = np.array(SECOND_PXL_list)
    # CURRENT_PXL_list = np.array(CURRENT_PXL_list)
    # steer_list = np.array(steer_list)
    # logged_data[columns[0]] = position_list
    # logged_data[columns[1]] = SECOND_PXL_list
    # logged_data[columns[2]] = CURRENT_PXL_list
    # logged_data[columns[3]] = steer_list
    # logged_data.to_excel('Output.xlsx')

    # np.save('Second.npy', SECOND_PXL_list)
    # np.save('Current.npy', CURRENT_PXL_list)
    # np.save('Steer.npy', steer_list)
    # np.save('Position.npy', position)
