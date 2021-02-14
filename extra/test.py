import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image', type = str, default = 'urban_views.png' , help = 'Choose the name of the image you want.')
args = parser.parse_args()
frame = cv2.imread(args.image)
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
plt.imshow(hsv_frame)
plt.show()