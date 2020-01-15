import cv2
import numpy as np
import datetime


basename = "img_"
subname = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
frmt = ".jpg"
index = 0


# Cam Init
cam = cv2.VideoCapture(0)
H = 480
W = 640
cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
#



while(1):
    ret, img = cam.read()
    img_calb = cv2.undistort(img, CamMat, DistMat, newCamMat);
    cv2.imshow('img',img)
    if(cv2.waitKey(10) == 27):
	    print("Start Record")
	    break

    
while(1):
    ret, img = cam.read()
    img_calb = cv2.undistort(img, CamMat, DistMat, newCamMat);
    cv2.imshow('img',img)
    
    filename = "".join([basename, subname, str(index),frmt])
    cv2.imwrite(filename,img, params=None)
    index = index+1
    if(cv2.waitKey(50) == 27):
	    print("Stop Record")
	    break
      
cam.release()
cv2.destroyAllWindows()
