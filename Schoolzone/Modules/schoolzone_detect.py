# import modules
import cv2
import numpy as np

# define functions
"""
description - line detect & return line array  -> It will be replaced to Cpp code
input
  image : bin image (gray scale, ipm)
  output
  line : left(0) & right(1) line array (np)
"""
def line_detect(image):
    #img = cv2.threshold(cv2.bitwise_not(cv2.flip(image,0)),150,1,cv2.THRESH_BINARY)
    H, W = imgage.shape
    pos_M = int(W/2)
    line_L = [0 for i in range(H)]
    line_R = [0 for i in range(H)]
    band = 10
    
    for j in range(0,H):
        if(j==0):
            list_L = range(int(pos_M*3/4),int(pos_M))
            list_R = range(int(pos_M),int(pos_M*5/4))
        else:
            if(line_L[j-1]-band < 0 ): LL = 0
            else: LL = line_L[j-1]-band
            if(line_R[j-1]+band > W): RR = W
            else: RR = line_R[j-1]+band
            list_L = range(LL,line_L[j-1]+band)
            list_R = range(line_R[j-1]-band,RR)
        
        line_L[j] = int(sum(img_result[j,list_L[0]:list_L[-1]+1]*list_L)/sum(img_result[j,list_L[0]:list_L[-1]+1]))
        line_R[j] = int(sum(img_result[j,list_R[0]:list_R[-1]+1]*list_R)/sum(img_result[j,list_R[0]:list_R[-1]+1]))
        img_cp[j+130,line_L[j]+int(orgW/2)-80] = 255
        img_cp[j+130,line_R[j]+int(orgW/2)-80] = 255
        
        pos_M = int((line_L[j]+line_R[j])/2)
        
    line =np.array([line_L,line_R])
    
    return line
    
    



"""
description - crop image bw lines
input
  image : original image (gray scale, ipm)
  line_left : left line x coord array
  line_right : right line x coord array
output
  image_roi : cropped image inter left & right line
"""
def roi(image, line_left, line_right):
    x1 = max(line_left)
    x2 = max(line_right)
    image_roi = image[:,x1:x2]
    
    return image_roi
