# @ 2020, Copyright Amirmohammad Zarif
# Compatible with firasimulator version 1.0.1 or higher
import FiraAuto
import time
import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import math

#Calling the class
car = FiraAuto.car()

#connecting to the server (Simulator)
car.connect("127.0.0.1", 25001)
car_mask = np.load('./car_mask.npy')

counter = 0
name_counter = 0

while(True):  
    counter = counter + 1
    car.setSteering(0)
    car.setSpeed(30)
    car.getData()
    if(counter > 4):

        sensors = car.getSensors() 
        frame = car.getImage()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, np.array([100,170,90]), np.array([160,220,220]))
        white_mask = cv2.inRange(frame, np.array([240,240,240]), np.array([255,255,255])) * (1-car_mask)
        points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            if points != None:
                sorted_points = sorted(points, key=len)
                if cv2.contourArea(sorted_points[-1])>25:
                    x,y,w,h = cv2.boundingRect(sorted_points[-1])
                    if (x>5) and (x+w<251) and (y>5) and (y+h<251):
                        sign = frame[y:y+h,x:x+w] 
                        print(sign.shape)  
                        sign = cv2.resize(sign, (25,25))
                        cv2.imwrite('./signs/sign_{}.png'.format(name_counter), sign)
                        print('saved number : ', name_counter)
                        name_counter += 1
                        time.sleep(0.2)
        except:
            pass

        key = cv2.waitKey(1)
        if key == ord('w'):
            cv2.imwrite('./obstacle_frame.jpg', frame)

    #A brief sleep to make sure everything 
