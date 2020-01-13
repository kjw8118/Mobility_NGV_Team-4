import cv2
import numpy as np
import math

alpha = (10-90)*np.pi/180
beta = (0)*np.pi/180
gamma = (4)*np.pi/180
dist = 300
f = 800

img_schoolzone = cv2.imread('C:\schoolzone.jpg', cv2.IMREAD_COLOR)

H,W = img_schoolzone.shape[0:2]

A1 = np.array([[1,0,-W/2],[0,1,-H/2],[0,0,0],[0,0,1]],dtype='f')
RX = np.array([[1,0,0,0],[0,math.cos(alpha), -math.sin(alpha),0],[0,math.sin(alpha),math.cos(alpha),0],[0,0,0,1]],dtype='f')
RY = np.array([[math.cos(beta),0,-math.sin(beta),0],[0,1,0,0],[math.sin(beta),0,math.cos(beta),0],[0,0,0,1]],dtype='f')
RZ = np.array([[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]],dtype='f')
R = np.dot(RX,np.dot(RY,RZ))

T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,dist],[0,0,0,1]],dtype='f')
K = np.array([[f,0,W/2,0],[0,f,H/2,0],[0,0,1,0]],dtype='f')

P=np.dot(K,np.dot(T,np.dot(R,A1)))

ipm = np.zeros((int(H),int(W),3),np.uint8)


V=int(H/2+f*math.tan(alpha))

ipm = cv2.warpPerspective(img_schoolzone,P,(int(W),int(H)), flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP)

img_gray =cv2.cvtColor(ipm,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 50, 150, apertureSize = 3)

img_crop = edges[0:H, int(W/2-100):int(W/2+100)]


cv2.imshow('org',img_schoolzone)
cv2.waitKey(0)

cv2.imshow('ipm',ipm)
cv2.waitKey(0)

cv2.imshow('edges',edges)
cv2.waitKey(0)
    
cv2.imshow('crop',img_crop)
cv2.waitKey(0)

cv2.destroyAllWindows()
