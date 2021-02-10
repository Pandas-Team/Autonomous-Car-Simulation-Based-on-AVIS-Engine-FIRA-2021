import FiraAuto
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import urban_utils as utils

car = FiraAuto.car()
car.connect("127.0.0.1", 25001)

# variables
REFRENCE = 128
CURRENT_PXL = 128

sign_state = 'nothing'

kp = 2
ki = 0.1
kd = 0.1
previous_error = 0
integral = 0
steer = 0
dt = 0.05
sensors = [1500,1500,1500]
# car_mask = cv2.imread('./car_mask.jpg',0)
# car_mask[car_mask<128] = 0
# car_mask[car_mask>=128] = 1
car_mask = np.load('./car_mask.npy')

# initializing
for _ in range(10):
    car.setSteering(0)
    car.setSpeed(10)
    car.getData()

# main loop
while(True):  
    # getting data 
    car.getData()
    sensors = car.getSensors() 
    frame = car.getImage()
    white_mask = cv2.inRange(frame, np.array([240,240,240]), np.array([255,255,255])) * (1-car_mask)
    side_mask = cv2.inRange(frame, np.array([130,0,108]), np.array([160,160,200])) * (1-car_mask)

    # vertical lines
    lines = utils.detect_lines(utils.region_of_interest(white_mask))
    two_line_mask, CURRENT_PXL = utils.mean_lines(white_mask, lines)

    # white horiz line
    horiz_detected = utils.horiz_lines(white_mask)

    # only on turns and obstacle
    mean_pix = utils.turn_where(white_mask)
    side_pix = utils.detect_side(side_mask)
    
    # detecting sign type
    sign = utils.detect_sign(frame, white_mask)
    if sign == 'left':
        sign_state = 'left'
    elif sign == 'straight':
        sign_state = 'straight'
    elif sign == 'right':
        sign_state = 'right'


    error = REFRENCE - CURRENT_PXL 
    steer = -(kp * error)
    car.setSteering(steer)
    car.setSpeed(30)

    if horiz_detected:
        car.setSteering(0)
        print(sign_state)
        ret = utils.stop_the_car(car)
        time.sleep(3)
        
        if sign_state == 'nothing':
            # turn based on mean_pix
            mean_pix = utils.turn_where(white_mask)
            print('mean_pix :', mean_pix)
            utils.go_back(car)
            if mean_pix < 128:
                utils.turn_the_car(car,-100,10)
            else:
                utils.turn_the_car(car,100,10)
            
        elif sign_state == 'left':
            utils.turn_the_car(car,-60,10)
            sign_state == 'nothing'

        elif sign_state == 'straight':
            utils.turn_the_car(car,0,12)
            sign_state == 'nothing'

        elif sign_state == 'right':
            utils.turn_the_car(car,60,10)
            sign_state == 'nothing'

        sign_state = 'nothing'

    if sensors[1] < 700:
        ret = utils.stop_the_car(car)
        print('side_pix :', side_pix)
        time.sleep(3)
        if side_pix > 128:
            utils.turn_the_car(car,-100,4)
        else:
            utils.turn_the_car(car,100,4) 
        


    # showing some info
    result = np.concatenate([np.concatenate([frame,cv2.cvtColor(utils.region_of_interest(white_mask)*255, cv2.COLOR_GRAY2BGR)], 0),
             np.concatenate([cv2.cvtColor(two_line_mask, cv2.COLOR_GRAY2BGR),cv2.cvtColor(side_mask, cv2.COLOR_GRAY2BGR)],0)],1)

    cv2.imshow('car\'s perception', result)   

    key = cv2.waitKey(1)
    if key == ord('w'):
        frame = car.getImage()
        cv2.imwrite('./output_frame.png', frame)
        # cv2.imwrite('./new_roi.jpg', utils.region_of_interest(white_mask)*255)

#A brief sleep to make sure everything 
    
