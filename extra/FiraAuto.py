# @ 2020, Copyright Amirmohammad Zarif
import cv2
import os
import io
import re
import time
import math
import base64
import socket
import numpy as np
from PIL import Image
from array import array

#Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

#convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

class car():
    steering_value = 0
    speed_value = 0
    sensor_status = 1
    image_mode = 1
    get_Speed = 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_arr = [speed_value,steering_value,image_mode,sensor_status,get_Speed]
    data_str = "Speed:" + str(data_arr[0]) + ",Steering:" + str(data_arr[1]) + ",ImageStatus:" + str(data_arr[2]) + ",SensorStatus:" + str(data_arr[3]) + ",GetSpeed:" + str(data_arr[4])
    image = None
    sensors = None
    current_speed = None
    def connect(self,server,port):
        try:
            self.sock.connect((server, port))
            self.sock.settimeout(5.0)
            print("connected to ", server, port)
            return True
        except:
            print("Failed to connect to ", server, port)
            return False
    def setSteering(self,steering):
        self.steering_value = steering
        self.image_mode = 0
        self.sensor_status = 0
        self.updateData()
        self.sock.sendall(self.data_str.encode("utf-8"))
        time.sleep(0.01)

    def setSpeed(self,speed):
        self.speed_value = speed
        self.image_mode = 0
        self.sensor_status = 0
        self.updateData()
        self.sock.sendall(self.data_str.encode("utf-8"))
        time.sleep(0.01)
    
    def move(self):
        self.updateData()
        self.sock.sendall(self.data_str.encode("utf-8"))
         
    def getData(self):
        self.image_mode = 1
        self.sensor_status = 1
        self.updateData()
        self.sock.sendall(self.data_str.encode("utf-8"))
        recive = self.sock.recv(80000).decode("utf-8")
        imageTagCheck = re.search('<image>(.*?)<\/image>', recive)
        sensorTagCheck = re.search('<sensor>(.*?)<\/sensor>', recive)
        speedTagCheck = re.search('<speed>(.*?)<\/speed>', recive)
        
        
        try:
            if(imageTagCheck):
                imageData = imageTagCheck.group(1)
                im_bytes = base64.b64decode(imageData)
                im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
                imageOpenCV = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
                self.image = imageOpenCV
            
            if(sensorTagCheck):
                sensorData = sensorTagCheck.group(1)
                sensor_arr = re.findall("\d+", sensorData)
                sensor_int_arr = list(map(int, sensor_arr)) 
                self.sensors = sensor_int_arr
            else:
                self.sensors = [1500,1500,1500]
            if(speedTagCheck):
                current_sp = speedTagCheck.group(1)
                self.current_speed = int(current_sp)
            else:
                self.current_speed = 0
            
            
        except:
            pass
            #print("Unvalid Recive!")
            

    def getImage(self):
        return self.image

    def getSensors(self):
        return self.sensors
    
    def getSpeed(self):
        return self.current_speed
    
    def updateData(self):
        data = [self.speed_value,self.steering_value,self.image_mode,self.sensor_status,self.get_Speed]
        self.data_str = "Speed:" + str(data[0]) + ",Steering:" + str(data[1]) + ",ImageStatus:" + str(data[2]) + ",SensorStatus:" + str(data[3]) + ",GetSpeed:" + str(data[4])
    
    def stop(self):
        self.setSpeed(0)
        self.setSteering(0)
        self.sock.sendall("stop".encode("utf-8"))
        self.sock.close()
        print("done")
    
    def __del__(self):
        self.stop()
    
        
        
