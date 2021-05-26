# @ 2020, Copyright Amirmohammad Zarif
# Compatible with firasimulator version 1.0.1 or higher
import AVISEngine
import time
import cv2
import numpy
import os

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok= True)
saved = 0
#Calling the class
car = AVISEngine.car()

#connecting to the server (Simulator)
car.connect("127.0.0.1", 25001)

#Counter variable
counter = 0

debug_mode = False
#sleep for 2 second to make sure that client connected to the simulator 
time.sleep(3)
try:
    while(True):
        #Counting the loops
        
        counter = counter + 1

        #Set the power of the engine the car to 20, Negative number for reverse move, Range [-100,100]
        # car.setSpeed(20)

        #Set the Steering of the car -10 degree from center
        # car.setSteering(-10)

        #Get the data. Need to call it every time getting image and sensor data
        car.getData()

        #Start getting image and sensor data after 4 loops. for unclear some reason it's really important 
        if(counter > 20):

            #returns a list with three items which the 1st one is Left sensor data, the 2nd one is the Middle Sensor data, and the 3rd is the Right one.
            sensors = car.getSensors() 
            image = car.getImage()
            carSpeed = car.getSpeed()
            # car.setSpeed(20)
            #showing the opencv type image
            cv2.imshow('frames', image)
            #break the loop when q pressed

            
            k = cv2.waitKey(10)
            if  k == ord('q'):
                break
            # elif k == ord('s'):
            #     saved += 1
            #     name = "out_{:03d}.npy".format(saved)
            #     name = os.path.join(output_dir, name)
            #     print('Saved')
            time.sleep(0.1)
        #A brief sleep to make sure everything 
        
finally:
    car.stop()





