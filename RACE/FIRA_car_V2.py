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
kp = 1
ki = 0
kd = 0

previous_error = 0
integral = 0
steer = 0
dt = 0.05
sensors = [1500,1500,1500]
position = 'right'
#sleep for 3 second to make sure that client connected to the simulator 
time.sleep(3)
time1 = time.time()
try:
    while(True):  
        if ((sensors[2]!=1500 or sensors[1]!=1500) and (position=='right')):
            error = REFRENCE - SECOND_PXL 

        elif ((sensors[0]!=1500 or sensors[1]!=1500) and (position=='left')):
            error = REFRENCE - SECOND_PXL 

        else:
            error = REFRENCE - CURRENT_PXL 

        integral = integral + error * dt    
        # if (ki * integral) > 10:
        #     integral = 10/ki
        derivative = (error - previous_error) / dt

        steer = -(kp * error + ki * integral + kd * derivative)
        # steer = -(kp * error)
        # print(steer)
        
        counter = counter + 1
        
        car.setSpeed(60)
        car.setSteering(steer)

        car.getData()

        if(counter > 4):
            sensors = car.getSensors() 
            frame = car.getImage()
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame = cv2.medianBlur(hsv_frame, 7)
            hsv_frame = cv2.medianBlur(hsv_frame, 5)
            hsv_frame = cv2.medianBlur(hsv_frame, 3)

            mask = cv2.inRange(hsv_frame, np.array([100,8,27]), np.array([117,50,52]))
            mask[0:110,:]=0
            points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_points = sorted(points, key=len)

            right_lane_mask = cv2.fillPoly(np.zeros((256,256)), pts =[sorted_points[-1]], color=(255))
            left_lane_mask = cv2.fillPoly(np.zeros((256,256)), pts =[sorted_points[-2]], color=(255))
            obstacle_mask = cv2.inRange(frame, np.array([70,4,93]), np.array([115,16,150]))
            # obstacle_mask2 = cv2.inRange(frame, np.array([104,88,77]), np.array([255,211,93]))
            obstacle_res = obstacle_mask
            # obstacle_res = cv2.bitwise_or(obstacle_mask, obstacle_mask2)
            
            yellow_mask = cv2.inRange(hsv_frame, np.array([28,115,154]), np.array([31,180,255]))


            points, _ = cv2.findContours(obstacle_res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_points = sorted(points, key=len)
            try:
                if cv2.contourArea(sorted_points[-1])>25:
                    x,y,w,h = cv2.boundingRect(sorted_points[-1])
                    mean_obstacle = np.mean(np.where(obstacle_res[y:y+h,x:x+w]>0), axis=1)[1] 

                    yellow_roi = np.mean(np.where(yellow_mask[y:y+h, :]>0), axis=1)[1] 

                    if mean_obstacle<yellow_roi:
                        print('obstacle is on left')
                    else:
                        print('obstacle is on right')

                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            except:
                pass
            

            CURRENT_PXL = np.mean(np.where(right_lane_mask[120:150,:]>0), axis=1)[1]
            SECOND_PXL = np.mean(np.where(left_lane_mask[120:150,:]>0), axis=1)[1]

            if np.isnan(CURRENT_PXL): CURRENT_PXL = 128
            if np.isnan(SECOND_PXL): SECOND_PXL = 128

            direction_s,slope = detect_yellow_line(frame)

            if slope<0:
                position = 'left'
            else:
                position = 'right'
            
            cv2.putText(direction_s, position, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # ImShow
            show_img = np.concatenate((frame, direction_s), axis=1)
            h1_axis = np.concatenate((left_lane_mask, right_lane_mask), axis=1)
            h2_axis = np.concatenate((obstacle_res, yellow_mask), axis=1)
            show_mask = np.concatenate((h1_axis, h2_axis), axis=0) 

            cv2.imshow('img',show_img)
            cv2.imshow('info',show_mask)
            key = cv2.waitKey(1)
            # if key == ord('w'):
            #     cv2.imwrite('./obstacle_frame.jpg', frame)
        previous_error = error
        # sleep(dt)
        os.system('cls')
        print(f'Current : {CURRENT_PXL}')
        print(f'Second : {SECOND_PXL}')
        print(f'Error : {error}')

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
