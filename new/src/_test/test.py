'''
Created on 2020. 2. 9.

@author: jinwon
'''
import cv2
#from main import Img


class test:
    def __init__(self,a):
        self.a = a
        


def putvar(test, b):
    test.a = b
    
A = test(1)

print(A.a)

putvar(A,2)

print(A.a)
img = cv2.imread('dog.png',cv2.IMREAD_COLOR)
img2 = cv2.imread('logo.png',cv2.IMREAD_UNCHANGED)
img2 = cv2.resize(img2, dsize=(100,100),interpolation=cv2.INTER_AREA)
print(img2.shape)
H,W = img2.shape[0:2]
print(img.shape)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
#cv2.imshow('b',img)
cv2.imshow('b',img1[:,:,3])
cv2.waitKey(0)
img1[20:(20+H),20:(20+W),:] = img2[:,:,:]
A[2:5,3:6]
cv2.imshow('a',img1[:,:,3])
cv2.waitKey(0)
"""
#img3 = cv2.add(img,img2)
cv2.namedWindow('a')
cv2.moveWindow('a',0,0)
cv2.namedWindow('b')
cv2.moveWindow('b',600,0)
#img = cv2.add(img,img2)
cv2.imshow('a',img)
cv2.imshow('b',img2)
cv2.waitKey(0)

img = cv2.subtract(img,img2)
cv2.imshow('a',img)
cv2.waitKey(0)
"""