import cv2
import numpy as np
import math
import glob

image = glob.glob('*.jpg')
print("Number of image :",len(image))
index = 1

# Calib
CamMat = np.array([[314.484, 0, 321.999],[0, 315.110, 259.722],[ 0, 0, 1]],dtype='f')
DistMat = np.array([ -0.332015,	0.108453,	0.001100,	0.002183],dtype='f')
#

H = 480
W = 640

# Gen Matrix
alpha = (8-90)*np.pi/180
beta = (0)*np.pi/180
gamma = (0)*np.pi/180
dist = 300
f=500
A1 = np.array([[1,0,-W/2],[0,1,-H/2],[0,0,0],[0,0,1]],dtype='f')
RX = np.array([[1,0,0,0],[0,math.cos(alpha), -math.sin(alpha),0],[0,math.sin(alpha),math.cos(alpha),0],[0,0,0,1]],dtype='f')
RY = np.array([[math.cos(beta),0,-math.sin(beta),0],[0,1,0,0],[math.sin(beta),0,math.cos(beta),0],[0,0,0,1]],dtype='f')
RZ = np.array([[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]],dtype='f')
R = np.dot(RX,np.dot(RY,RZ))
T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,dist],[0,0,0,1]],dtype='f')
K = np.array([[f,0,W/2,0],[0,f,H/2,0],[0,0,1,0]],dtype='f')
P=np.dot(K,np.dot(T,np.dot(R,A1)))
img_ipm = np.zeros((int(H),int(W),3),np.uint8)
#

c_calib = "calib_"
c_ipm = "ipm_"

print("\nPost-Processing...")
for fname in image:
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img_calib = cv2.undistort(img, CamMat, DistMat, None, CamMat)
    img_ipm = cv2.warpPerspective(img_calib,P,(int(W),int(H)), flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP)
    filename_calib = "".join([c_calib, fname])
    filename_ipm = "".join([c_ipm, fname])
    cv2.imwrite(filename_calib,img_calib, params=None)
    cv2.imwrite(filename_ipm,img_ipm, params=None)
    print("[",index,"] : ",fname," is done.")
    index = index+1
    
print("\n\nAll image post-processing is Done")
