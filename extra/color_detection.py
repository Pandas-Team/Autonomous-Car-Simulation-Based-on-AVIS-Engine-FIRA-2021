import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image', type = str, default = 'urban_views.png' , help = 'Choose the name of the image you want.')
args = parser.parse_args()


def nothing(x):
    pass

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
trackwindow = (0,0,240,320)

cap = cv2.VideoCapture(0)

cv2.namedWindow('trackbars')
cv2.createTrackbar('high H : ', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('high S : ', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('high V : ', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('low H : ', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('low S : ', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('low V : ', 'trackbars', 0, 255, nothing)

while True:
    high_h = cv2.getTrackbarPos('high H : ', 'trackbars')
    high_s = cv2.getTrackbarPos('high S : ', 'trackbars')
    high_v = cv2.getTrackbarPos('high V : ', 'trackbars')
    low_h = cv2.getTrackbarPos('low H : ', 'trackbars')
    low_s = cv2.getTrackbarPos('low S : ', 'trackbars')
    low_v = cv2.getTrackbarPos('low V : ', 'trackbars')

    upper = np.array([high_h, high_s, high_v])
    lower = np.array([low_h, low_s, low_v])

    frame = cv2.imread(args.image)
    hsv_frame = frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv_frame = cv2.blur(hsv_frame,(6,6))
    mask = cv2.inRange(hsv_frame, lower, upper)
    # mask = cv2.GaussianBlur(mask , (7,7), 0)
    # mask = cv2.medianBlur(mask , 7)
    result = cv2.bitwise_and(frame, frame, mask = mask)


    #detecting using contours
    points,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, points, -1, (0,0,255), 2)

    # plotting ROI
    result = cv2.rectangle(result,(0,140),(512,190),(0,255,255),1)
    what_to_show = np.vstack((frame, result))
    print(frame.shape, result.shape, mask.shape)
    # cv2.imshow('frame',frame)
    # cv2.imshow('result',result)
    cv2.imshow('mask', mask)
    # cv2.imshow('frame',frame[150:190, :])
    # cv2.imshow('mask', mask[150:190, :])
    cv2.imshow('frame + result + mask', what_to_show)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('w'):
        cv2.imwrite('./frame2.jpg', mask)


cap.release()
cv2.destroyAllWindows()