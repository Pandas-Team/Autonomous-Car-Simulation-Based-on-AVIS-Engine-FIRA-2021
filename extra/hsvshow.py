import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt 
parser = argparse.ArgumentParser()
parser.add_argument('--image', type = str, default = 'race_views.png' , help = 'Choose the name of the image you want.')
args = parser.parse_args()

if 'npy' in args.image:
    frame = np.load(args.image)
else:
    frame = cv2.imread(args.image)

hsv_frame = frame
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hsv_frame = cv2.medianBlur(hsv_frame, 7)
# hsv_frame = cv2.medianBlur(hsv_frame, 5)
plt.imshow(hsv_frame)
plt.show()