import cv2
import numpy as np
import math


cam = cv2.VideoCapture(-1) # For Raspberry-pi :-1 / Laptop : 0

W = cam.get(3)
H = cam.get(4)

alpha = (28-90)*np.pi/180
beta = 0
gamma = 0
f = 500
dist = 500

A1 = np.array([[1,0,-W/2],[0,1,-H/2],[0,0,0],[0,0,1]],dtype='f')
RX = np.array([[1,0,0,0],[0,math.cos(alpha), -math.sin(alpha),0],[0,math.sin(alpha),math.cos(alpha),0],[0,0,0,1]],dtype='f')
RY = np.array([[math.cos(beta),0,-math.sin(beta),0],[0,1,0,0],[math.sin(beta),0,math.cos(beta),0],[0,0,0,1]],dtype='f')
RZ = np.array([[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]],dtype='f')
R = np.dot(np.dot(RX,RY),RZ)

T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,dist],[0,0,0,1]],dtype='f')
K = np.array([[f,0,W/2,0],[0,f,H/2,0],[0,0,1,0]],dtype='f')

P=np.dot(np.dot(np.dot(K,T),R),A1)

ipm = np.zeros((int(H),int(W),3),np.uint8)

while(1):
    ret, frame = cam.read()
    ipm = cv2.warpPerspective(frame,P,(int(W),int(H)))
    cv2.imshow('ipm',ipm)
    if(cv2.waitKey(10)==27): break


cam.release()
cv2.destroyAllWindows()
