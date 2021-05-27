# @ 2020, Copyright Amirmohammad Zarif
# Compatible with firasimulator version 1.0.1 or higher
import AVISEngine
import time
import cv2
import numpy as np
from functions import *
from time import sleep
import pandas as pd
#Calling the class
car = AVISEngine.car()
np.seterr(all="ignore")
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
kp = 1.2
ki = 0.5
kd = 0
angle = 0
previous_error = 0
integral = 0.0
steer = 0
dt = 0.05
sensors = [1500,1500,1500]
position = 'right'
#sleep for 3 second to make sure that client connected to the simulator 
time.sleep(3)
time1 = time.time()
error = 0
MIDDLE_RED = 140 
where_yellow = 1 # 1 for right, 0 for left
where_white = -1
epsilon = 1e-5
where_avg = 0.1
pos = 3
sensors_array = np.array([1500,1500,1500])
car_mode = 1 #right and clear

try:
    while(True):  

        # error = - angle
        # integral = integral + error * dt    
        # if (ki * integral) > 10:
        #     integral = 10/ki
        # derivative = (error - previous_error) / dt

        # steer = (kp * error + ki * integral + kd * derivative)

        steer = (kp * error)
        # if np.round(pos) == 1:
        #     error = (2 - pos ) * 30
        #     steer = (kp * error)
        # elif np.round(pos) == 4:
        #     steer = -40
        if (car_mode == 1) :#or (car_mode == 3):
            turn_right_error = 40
            steer = (kp * error)

        elif (car_mode == 2):
            turn_right_error = 40
            error =  - 1.3 * (1500 - sensors_array[1]) + 0.5 * (1500 - sensors_array[2])
            steer =  (0.1 * error)
        elif (car_mode == 3):
            turn_right_error = turn_right_error - 1
            error = turn_right_error
            steer = (kp * error)
        elif (np.round(pos) == 1):
            steer = (+20)



        # else: 
        #     error = 3000 - sensors_array[1] - sensors_array[2] 
        #     steer = - (0.05 * error) 

        # elif (pos < 2.5 or (car_mode is None)) :
        #     steer = +45
        # elif car_mode == 2:
        #     steer = -45
        # 

        # if (np.round(pose) == 3): #or (car_mode == 3):
        #     steer = (kp * error)
        # if car_mode == 2:
        #     steer = -30
        # if np.round(pose) == 3 or (car_mode is None):
        #     steer = +30
        # print(steer)
        
        counter = counter + 1
        
        car.setSpeed(100)
        car.setSteering(steer)

        car.getData()

        if(counter > 4):
            sensors = car.getSensors() 
            frame = car.getImage()
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame = cv2.medianBlur(hsv_frame, 7)

            obstacle_mask = cv2.inRange(frame, np.array([70,4,93]), np.array([115,16,150]))
            
            # obstacle_mask2 = cv2.inRange(frame, np.array([104,88,77]), np.array([255,211,93]))
            obstacle_res = obstacle_mask
            # obstacle_res = cv2.bitwise_or(obstacle_mask, obstacle_mask2)
            
            sensors_array = where_avg * (np.array(sensors)) + (1-where_avg) * sensors_array
            sensors_array_rounded = np.round(sensors_array , -2)
            yellow_mask = cv2.inRange(hsv_frame, np.array([28,115,154]), np.array([31,180,255]))
            yellow_left =  yellow_mask[:,:128]
            yellow_right = yellow_mask[:,128:]
            yellow_left_score =   yellow_left.mean()
            yellow_right_score =  yellow_right.mean()
            where_yellow =  where_avg * (yellow_right_score - yellow_left_score ) / (yellow_left_score + yellow_right_score + epsilon) + (1-where_avg) * where_yellow
            where_yellow = np.nan_to_num(where_yellow, nan = 1)

            white_line_mask = cv2.inRange(hsv_frame, np.array([0, 0, 210]), np.array([51,18,255]))
            white_line_mask[0:100] = 0 #Apply ROI
            white_left = white_line_mask[:,:128]
            white_right = white_line_mask[:,128:]
            white_left_score = white_left.mean()
            white_right_score = white_right.mean()
            where_white =  where_avg * (white_right_score - white_left_score ) / (white_left_score + white_right_score + epsilon) + (1-where_avg) * where_white
            where_white = np.nan_to_num(where_white)
            pos = where_avg * find_position(where_white, where_yellow)  + (1-where_avg) * pos

            lane_mask = cv2.bitwise_or(yellow_mask, white_line_mask)
            yellow_test = np.mean(np.where(yellow_mask[120:150,:]>0), axis=1)[1]
            white_test = np.mean(np.where(white_line_mask[120:150,:]>0), axis=1)[1]
            yellow_test = np.nan_to_num(yellow_test)
            white_test = np.nan_to_num(white_test)

            middle_test = (yellow_test + white_test) / 2
            error = middle_test - MIDDLE_RED


      
            lane_concat = np.concatenate((white_line_mask, yellow_mask))

            # cv2.imshow('img',show_img)
            # cv2.imshow('info',show_mask)
            cv2.imshow('Lane Mask', lane_concat)
            cv2.imshow("Lane mask 2 ", lane_mask)
            key = cv2.waitKey(1)

        try :

            os.system('cls')
            print(f'Counter : {counter}')
            print(f'Where Yellow : {np.round(where_yellow , 2)}')
            print(f'Where White : {np.round(where_white, 2)}')
            print(f'Actual Where : {np.round(pos,2)}')
            # print(f'Yellow Test : {yellow_test}')
            # print(f'White Test : {white_test}')
            # print(f'Middle Test : {middle_test}')
            print(f'Steer : {steer}')
            car_mode = car_status(pos, sensors_array_rounded)
            print(f"Car Mode : ", car_mode)
            print(sensors_array_rounded)
            # print(f'white_left_score : {white_left_score}')
            # print(f'white_right_score : {white_right_score}')
            # print(f'yellow_left_score : {yellow_left_score}')
            # print(f'yellow_right_score : {yellow_right_score}')
        except: pass

        # print()
        
        try:
            img_where = np.argwhere(yellow_mask)
            x = img_where[:, 1]
            y = 255 - img_where[:, 0]
            out = interval_avg(x,y)
            out[0] = out[0][~np.isnan(out[0])]
            out[1] = out[1][~np.isnan(out[1])]

            dx = np.gradient(out[0])
            dy = np.gradient(out[1])
            dy_dx = dy / dx
            angles = np.rad2deg(np.arctan(dy_dx))
            angle = 55 - np.mean(angles)
            # angle = angles[-1] - angles[0]
            print(f'Angle : {angle}')

        except : 
            angle = angle        
        
finally:

    car.stop()

