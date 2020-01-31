import ctypes
from ctypes import *
import cv2
import numpy as np

c_module = cdll.LoadLibrary('./detectmodule_cpp.dll')

img = cv2.imread('ipm_img.jpg')
H, W, D = img.shape
#img2 = np.zeros(img.shape, dtype='uint8')
pimg = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
print(pimg)
img2 = np.ctypeslib.as_array(pimg, shape=img.shape)
print(img2.shape)
cv2.imshow('1',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
parr = c_module.line_detect(pimg, H, W)




c_module.line_detect_free(img,parr)
