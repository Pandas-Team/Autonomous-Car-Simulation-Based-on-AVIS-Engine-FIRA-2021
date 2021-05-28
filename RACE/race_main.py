# @ 2020, Copyright Amirmohammad Zarif
# Compatible with firasimulator version 1.0.1 or higher
import AVISEngine
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from race_utils import *
from time import sleep
np.seterr(all="ignore")
def lane_change_to_right():
    time1 = time.time()
    while((time.time()-time1)<0.8):
        car.getData()
        car.setSteering(+50)
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
kp = 1
ki = 0.2
kd = 0.2
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
        if(time.time()-time1)>2:
            if (position=='left'):
                lane_change_to_right()   
        print(sensors)
        if ((sensors[2]<1500 or sensors[1]<1500) and (position=='right') and (mean_obstacle>yellow_roi) ):
            error = (REFRENCE - SECOND_PXL)
            time1 = time.time()

        elif ((sensors[0]<1500 or sensors[1]<1500) and (position=='left') and (mean_obstacle>yellow_roi) ):
            error = (REFRENCE - SECOND_PXL)
            time1 = time.time()
        else:
            error = REFRENCE - CURRENT_PXL 

        steer = -(kp * error)        
        counter = counter + 1
        
        car.setSpeed(60)
        car.setSteering(int(steer))

        car.getData()

        if(counter > 4):
            # getting data
            sensors = car.getSensors() 
            frame = car.getImage()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame = cv2.medianBlur(hsv_frame, 7)

            mask = cv2.inRange(frame, np.array([0,0,0]), np.array([60,45,40]))
            mask = cv2.medianBlur(mask, 5)
            cv2.imshow('lane mask', mask)
            # mask[0:110,:]=0
            points, _ = cv2.findContours(mask[150:200,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = []
            for i in range(len(points)):
                areas.append(cv2.drawContours(np.zeros((60,256)), points, i, 255, -1))
            areas = np.array(areas)
            sorted_areas = np.argsort(areas.sum(axis = (1,2)))
            try:
                right_lane_mask = areas[sorted_areas[-1],:,:]
                left_lane_mask = areas[sorted_areas[-2],:,:]
            except:
                pass
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
            

            CURRENT_PXL = np.mean(np.where(right_lane_mask>0), axis=1)[1]
            SECOND_PXL = np.mean(np.where(left_lane_mask>0), axis=1)[1]

            if np.isnan(CURRENT_PXL): CURRENT_PXL = 128
            if np.isnan(SECOND_PXL): SECOND_PXL = 128


            direction_s,slope = detect_yellow_line(frame)

            if slope<0:
                position = 'left'
            else:
                position = 'right'

            yellow_mask = cv2.inRange(hsv_frame, np.array([28,115,154]), np.array([31,180,255]))
            YELLOW_PXL = np.mean(np.where(yellow_mask[140:190,:]>0), axis=1)[1]
            if np.isnan(YELLOW_PXL): YELLOW_PXL = 128
            print(YELLOW_PXL)
            if YELLOW_PXL>128:
                position = 'left'
            else:
                position = 'right'

            cv2.putText(direction_s, position, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame,(0,150),(256,200),(0,255,255),1)
            # ImShow
            show_img = np.concatenate((frame, direction_s), axis=1)
            h1_axis = np.concatenate((left_lane_mask, right_lane_mask), axis=1)
            h2_axis = np.concatenate((obstacle_res, yellow_mask), axis=1)
            show_mask = np.concatenate((h1_axis, h2_axis), axis=0) 

            cv2.imshow('img',show_img)
            cv2.imshow('info',show_mask)
            key = cv2.waitKey(1)
            if key == ord('w'):
                np.save('masks.npy', show_mask)
                cv2.imwrite('./race_frame.png', frame)
        previous_error = error
        # sleep(dt)
        os.system('cls')
        print(f'Current : {CURRENT_PXL}')
        print(f'Second : {SECOND_PXL}')
        print(f'Error : {error}')
        print(f'Mean Obstacle : {mean_obstacle}')
        print(position)

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
