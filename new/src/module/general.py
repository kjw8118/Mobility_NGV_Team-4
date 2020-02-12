'''
Created on 2020. 2. 9.

@author: jinwon
'''
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from multiprocessing import Pool

class ImgProc:
    from multiprocessing import Process
    from threading import Thread
    import cv2
    import numpy as np
    import math
    
    def __init__(self, resolution, flag, imageset):
            
            
        
        self.H, self.W = [300, 600]
        self.schoolzone_t = self.np.full((self.H,self.W,3),0,dtype='uint8')
        self.schoolzone_t[:,:,2] = 255
        self.schoolzone_f = self.np.full((self.H,self.W,3),255,dtype='uint8')
        self.stopline_t = self.cv2.resize(self.cv2.imread('dashboard/stopline_red.jpg',self.cv2.IMREAD_COLOR), dsize=(int(self.W/2), self.H), interpolation=self.cv2.INTER_AREA)
        self.stopline_f = self.schoolzone_t[:,0:int(self.W/2)].copy()
        self.blindspot_t = self.cv2.resize(self.cv2.imread('dashboard/blind_spot_red.jpg',self.cv2.IMREAD_COLOR), dsize=(int(self.W/2), self.H), interpolation=self.cv2.INTER_AREA)
        self.blindspot_f = self.schoolzone_t[:,0:int(self.W/2)].copy()
        self.dashboard = self.np.full((self.H,self.W,3),255,dtype='uint8')
        

        self.imageset = imageset
        self.flag = flag
        self.prop_height, self.prop_width = resolution
        self.prop_CamMat = self.np.array([[314.484, 0, 321.999],[0, 315.110, 259.722],[ 0, 0, 1]],dtype='f')
        self.prop_DistMat = self.np.array([ -0.332015,    0.108453,    0.001100,    0.002183],dtype='f')
        
        self.prop_alpha = (5-90)*self.np.pi/180
        self.prop_beta = (0)*self.np.pi/180
        self.prop_gamma = (0)*self.np.pi/180
        self.prop_dist = 160
        self.prop_focal = 400
        
        A1 = self.np.array([[1,0,-self.prop_width/2],[0,1,-self.prop_height/2],[0,0,0],[0,0,1]],dtype='f')
        RX = self.np.array([[1,0,0,0],[0,self.math.cos(self.prop_alpha), -self.math.sin(self.prop_alpha),0],[0,self.math.sin(self.prop_alpha),self.math.cos(self.prop_alpha),0],[0,0,0,1]],dtype='f')
        RY = self.np.array([[self.math.cos(self.prop_beta),0,-self.math.sin(self.prop_beta),0],[0,1,0,0],[self.math.sin(self.prop_beta),0,self.math.cos(self.prop_beta),0],[0,0,0,1]],dtype='f')
        RZ = self.np.array([[self.math.cos(self.prop_gamma),-self.math.sin(self.prop_gamma),0,0],[self.math.sin(self.prop_gamma),self.math.cos(self.prop_gamma),0,0],[0,0,1,0],[0,0,0,1]],dtype='f')
        R = self.np.dot(RX,self.np.dot(RY,RZ))
        T = self.np.array([[1,0,0,0],[0,1,0,0],[0,0,1,self.prop_dist],[0,0,0,1]],dtype='f')
        K = self.np.array([[self.prop_focal,0,self.prop_width/2,0],[0,self.prop_focal,self.prop_height/2,0],[0,0,1,0]],dtype='f')
        self.prop_P = self.np.dot(K,self.np.dot(T,self.np.dot(R,A1)))
        
        
        self.prop_lower_k = self.np.array([0,0,0])
        self.prop_upper_k = self.np.array([180,255,100])
        self.prop_lower_r = self.np.array([150,0,0])
        self.prop_upper_r = self.np.array([180,255,255])
        
        self.prop_ROI_W = int(self.prop_width*0.1)*2
        self.prop_ROI_H = int(self.prop_height*0.3)        
        self.prop_ROI_far_rngH = range(int(self.prop_height*0.6 - self.prop_ROI_H),int(self.prop_height*0.6))
        self.prop_ROI_far_rngW = range(int(self.prop_width/2 - self.prop_ROI_W/2),int(self.prop_width/2 + self.prop_ROI_W/2))
        
        self.prop_ROI_near_rngH = range(int(self.prop_height - self.prop_ROI_H),int(self.prop_height))
        self.prop_ROI_near_rngW = range(int(self.prop_width/2 - self.prop_ROI_W/2),int(self.prop_width/2 + self.prop_ROI_W/2))
        
        self.prop_lane_list = self.np.array(range(0, self.prop_ROI_W))
        self.lane = self.np.full((self.prop_ROI_H,1), int(self.prop_ROI_W/2), dtype='uint32')
        self.laneL = self.np.full((self.prop_ROI_H,1), int(self.prop_ROI_W/2), dtype='uint32')
        self.laneR = self.np.full((self.prop_ROI_H,1), int(self.prop_ROI_W/2), dtype='uint32')
        self.lane_base = self.np.array(range(0,self.prop_ROI_W))
        #self.lane_err = 0
        
        if(self.imageset.usage == False):
            self.cam = self.cv2.VideoCapture(-1)
            self.cam.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.prop_width)
            self.cam.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.prop_height)

        self.freq = self.cv2.getTickFrequency()
        self.flag.schoolzone = False
        
        self.distance_mode = 0

        self.initCoral("Model")
        
    def getFrame(self):
        if(self.imageset.usage == True ):
            self.frame = self.imageset.callImage()
        else:
            ret, self.frame = self.cam.read()
            del(ret)
        #print(self.imageset.imageFname)
        #self.cv2.imshow('a',self.frame)
        #self.cv2.waitKey(0)
        
    def getCalibration(self):
        frame = self.frame
        self.calibration = frame.copy()#self.cv2.undistort(frame, self.prop_CamMat, self.prop_DistMat, None, self.prop_CamMat)
    
    def getMaskBG(self):
        white = self.np.full((self.prop_height,self.prop_width,3),255,dtype='uint8')
        white_ipm = self.cv2.warpPerspective(white, self.prop_P, (self.prop_width,self.prop_height), flags=self.cv2.INTER_CUBIC|self.cv2.WARP_INVERSE_MAP)
        self.prop_maskBG = self.cv2.bitwise_not(white_ipm)
         
    def getIPM(self):
        frame = self.calibration
        ipm1 = self.cv2.warpPerspective(frame, self.prop_P, (self.prop_width,self.prop_height), flags=self.cv2.INTER_CUBIC|self.cv2.WARP_INVERSE_MAP)
        ipm2 = self.cv2.add(ipm1, self.prop_maskBG)
        ipm3 = self.cv2.bilateralFilter(ipm2, 9, 50, 50)
        self.ipm = ipm3 #[int(self.prop_height - self.prop_ROI_H):int(self.prop_height),int(self.prop_width/2 - self.prop_ROI_W/2):int(self.prop_width/2 + self.prop_ROI_W/2)]
    def getMask(self):
        frame = self.ipm
        self.hsv =  self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        mask_r = self.cv2.inRange(self.hsv, self.prop_lower_r, self.prop_upper_r)
        mask_k = self.cv2.inRange(self.hsv, self.prop_lower_k, self.prop_upper_k)
        
        #mask_r_far = mask_r[self.prop_ROI_far_rngH, self.prop_ROI_far_rngW]
        self.mask_r = mask_r[self.prop_ROI_near_rngH, self.prop_ROI_near_rngH]
        
        if(self.distance_mode == 0):
            self.mask_k = mask_k[self.prop_ROI_far_rngH, self.prop_ROI_far_rngW]
            self.mask_k_lane = mask_k[self.prop_ROI_near_rngH, self.prop_ROI_near_rngH]
        else:
            self.mask_k_lane = mask_k[self.prop_ROI_far_rngH, self.prop_ROI_far_rngW]
            self.mask_k = mask_k[self.prop_ROI_near_rngH, self.prop_ROI_near_rngH]
        
        
        
    def detectSchoolzone(self):
        #print(self.np.sum(self.mask_r)/255,((self.prop_ROI_H)*(self.prop_ROI_W)*0.2))
        if((self.np.sum(self.mask_r)/255) > ((self.prop_ROI_H)*(self.prop_ROI_W)*0.2)):
            self.flag.schoolzone = True
        else:
            self.flag.schoolzone = False
            
    def closeCamera(self):
        self.cam.release()

    
    def laneDetect(self):

        frame = self.cv2.flip(self.mask_k_lane.copy(),0)
        
        H,W = self.prop_ROI_H, self.prop_ROI_W
        
        self.lane = self.np.zeros((H,1),dtype='uint32')
        laneL = np.zeros((H2,1),dtype='uint16')
        laneR = np.zeros((H2,1),dtype='uint16')
        
        self.lane = self.np.full((H,1), int(W/2), dtype='uint32')
        self.laneL = self.np.full((H,1), int(W/2), dtype='uint32')
        self.laneR = self.np.full((H,1), int(W/2), dtype='uint32')
        
        
        num0 = self.np.sum(frame[0,:] != False)
        if(num0 != 0): self.lane[0] = int(self.np.sum(self.lane_base*frame[0,:])/(255*num0))
        else: self.lane[0] = int(W/2)
        
        for j in range(1,H):
            rangeL = range(0, int(self.lane[j-1]))
            rangeR = range(int(self.lane[j-1]), int(W))
            numL = self.np.sum(frame[j,rangeL] !=  False)
            numR = self.np.sum(frame[j,rangeR] !=  False)
            if(numL == 0)|(numR == 0): self.lane[j] = self.lane[j-1]
            else:
                self.laneL[j] = self.np.sum(self.lane_base[rangeL]*frame[j,rangeL])/(255*numL)
                self.laneR[j] = self.np.sum(self.lane_base[rangeR]*frame[j,rangeR])/(255*numR)
                self.lane[j] = (self.laneR[j] + self.laneL[j])/2
            #frame[j,int(self.lane[j])] = 255
        self.flag.lane_err = (self.np.mean(lane)*2/W-1)
        

  
    
    import os


    
    def initCoral(self, modeldir="Model"):
        
        CWD_PATH =self.os.getcwd()
        MODEL_NAME = modeldir
        GRAPH_NAME = "edgetpu.tflite"
        LABELMAP_NAME = "labelmap.txt"
        # path to 
        PATH_TO_CKPT = self.os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
        PATH_TO_LABELS = self.os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
        
        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            self.coral_labels = [line.strip() for line in f.readlines()]
        
        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.coral_labels[0] == '???':
            del(self.coral_labels[0])
        
        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        self.coral_interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
                                  
        self.coral_interpreter.allocate_tensors()
        
        # Get model details
        self.coral_input_details = self.coral_interpreter.get_input_details()
        self.coral_output_details = self.coral_interpreter.get_output_details()
        self.coral_height = self.coral_input_details[0]['shape'][1]
        self.coral_width = self.coral_input_details[0]['shape'][2]

        self.coral_input_mean = 127.5
        self.coral_input_std = 127.5
        
        
        
    def inference(self):
        image = self.mask_k.copy()
        self.coral_imH, self.coral_imW = image.shape[0:2]
        
        # Get image & general
        self.coral_frame = self.cv2.cvtColor(image, self.cv2.COLOR_GRAY2RGB)
        frame_rgb = self.cv2.cvtColor(self.coral_frame, self.cv2.COLOR_BGR2RGB)
        frame_resized = self.cv2.resize(frame_rgb, (self.coral_width, self.coral_height))
        input_data = self.np.expand_dims(frame_resized, axis=0)


        # Perform the actual detection by running the model with the image as input
        self.coral_interpreter.set_tensor(self.coral_input_details[0]['index'],input_data)
        self.coral_interpreter.invoke()

        # Retrieve detection results
        self.coral_boxes = self.coral_interpreter.get_tensor(self.coral_output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        self.coral_classes = self.coral_interpreter.get_tensor(self.coral_output_details[1]['index'])[0] # Class index of detected objects
        self.coral_scores = self.coral_interpreter.get_tensor(self.coral_output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        
        
    def detect(self,threshold):
        
        self.flag.stopline = False
        # Threshold
        self.coral_min_conf_threshold = threshold
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(self.coral_scores)):
            if ((self.coral_scores[i] > self.coral_min_conf_threshold) and (self.coral_scores[i] <= 1.0)):
                if(self.coral_labels[int(self.coral_classes[i])] == "stopline"): self.flag.stopline = True
                
                    
    def integralPreproc(self):
        if(self.imageset.usage == True ):
            frame = self.imageset.callImage()
        else:
            ret, frame = self.cam.read()
            del(ret)
        
        #calibration = self.cv2.undistort(frame, self.prop_CamMat, self.prop_DistMat, None, self.prop_CamMat)
        calibration = self.cv2.resize(frame, dsize=(320,240), interpolation=self.cv2.INTER_AREA)
        #mask_k = frame.copy()
        #self.cv2.imshow('c',calibration)
        #calibration = frame.copy()
        ipm1 = self.cv2.warpPerspective(calibration, self.prop_P, (self.prop_width,self.prop_height), flags=self.cv2.INTER_CUBIC|self.cv2.WARP_INVERSE_MAP)
        ipm2 = self.cv2.add(ipm1, self.prop_maskBG)
        ipm3 = self.cv2.bilateralFilter(ipm2, 9, 50, 50)
        #print(self.prop_height, self.prop_ROI_H, int((self.prop_height - self.prop_ROI_H)))
        ipm = ipm3[int((self.prop_height - self.prop_ROI_H)):int((self.prop_height)),int((self.prop_width/2 - self.prop_ROI_W/2)):int((self.prop_width/2 + self.prop_ROI_W/2))]
        #ipm = calibration
        #ipm = ipm1
        hsv =  self.cv2.cvtColor(ipm, self.cv2.COLOR_BGR2HSV)
        self.mask_k = self.cv2.inRange(hsv, self.prop_lower_k, self.prop_upper_k)
        self.mask_r = self.cv2.inRange(hsv, self.prop_lower_r, self.prop_upper_r)
        #print(mask_k.shape)
        #self.mask_k = mask_k.copy()
        #self.cv2.imshow('b',self.mask_r)
        
    def taskPreproc(self):
        while(1):
            t1 = self.cv2.getTickCount()
            
            #self.integralPreproc()
            
            self.getFrame()
            self.getCalibration()
            self.getIPM()
            self.getMask()
            
            self.detectSchoolzone()
            self.laneDetect()
            
            print(self.flag.lane_err)
            
            if(self.flag.schoolzone == True):
                self.inference()
                self.detect(0.95)
            
            
            if(self.flag.schoolzone == True):
                self.addSchoolzone()
                if(self.flag.stopline == True):
                    self.addStopline()
                else:
                    self.subsStopline()
                if(self.flag.blindspot == True):
                    self.addBlindspot()
                else:
                    self.subsBlindspot()
            else:
                self.subsSchoolzone()
#            self.cv2.imshow('a',self.mask_k)
            self.show()
            
            #print(self.mask_k.shape)

            t2 = self.cv2.getTickCount()
            print(self.freq/(t2-t1))
            #print(self.flag.schoolzone, self.flag.stopline)
    def startPreproc(self):
        #self.getMaskBG
        #lists = ['1','2','3']
        #pool = Pool(processes=3)
        #prs = pool.map(self, lists)
        #pool.close()
        
        
        self.threadPreproc = self.Thread(target=self.taskPreproc)
        self.threadPreproc.start()
        
    def taskObjectDetect(self):
        while(1):
            #if(self.flag.schoolzone == True):
                t1 = self.cv2.getTickCount()
                self.inference()
                self.detect(0.95)
                t2 = self.cv2.getTickCount()
                print(self.freq/(t2-t1))
            #print(self.flag.schoolzone, self.flag.stopline)

    def startObjectDetect(self):
        #self.initCoral()
        self.threadObjectDetect = self.Process(target=self.taskObjectDetect)
        self.threadObjectDetect.start()
        

        
        

        """
class Dashboard:
    from multiprocessing import Process
    from threading import Thread
    import cv2
    import numpy as np
    
    def __init__(self, flag):
        
        self.H, self.W = [300, 600]
        self.schoolzone_t = self.np.full((self.H,self.W,3),0,dtype='uint8')
        self.schoolzone_t[:,:,2] = 255
        self.schoolzone_f = self.np.full((self.H,self.W,3),255,dtype='uint8')
        self.stopline_t = self.cv2.resize(self.cv2.imread('dashboard/stopline_red.jpg',self.cv2.IMREAD_COLOR), dsize=(int(self.W/2), self.H), interpolation=self.cv2.INTER_AREA)
        self.stopline_f = self.schoolzone_t[:,0:int(self.W/2)].copy()
        self.blindspot_t = self.cv2.resize(self.cv2.imread('dashboard/blind_spot_red.jpg',self.cv2.IMREAD_COLOR), dsize=(int(self.W/2), self.H), interpolation=self.cv2.INTER_AREA)
        self.blindspot_f = self.schoolzone_t[:,0:int(self.W/2)].copy()
        self.dashboard = self.np.full((self.H,self.W,3),255,dtype='uint8')
        self.flag = flag
        """
    def show(self):
        self.cv2.imshow('Dashboard', self.dashboard)
        self.cv2.waitKey(1)
        
    def addSchoolzone(self):
        self.dashboard = self.schoolzone_t.copy()
        #self.cv2.imshow('c',self.schoolzone_t)
        #self.cv2.imshow('d',self.dashboard)
        #self.cv2.waitKey(0)
        
    def subsSchoolzone(self):
        self.dashboard = self.schoolzone_f.copy()
        
    def addStopline(self):
        self.dashboard[:,0:int(self.W/2)] = self.stopline_t.copy()
        
    def subsStopline(self):
        self.dashboard[:,0:int(self.W/2)] = self.stopline_f.copy()
        
    def addBlindspot(self):
        self.dashboard[:,int(self.W/2):self.W] = self.blindspot_t.copy()
        
    def subsBlindspot(self):
        self.dashboard[:,int(self.W/2):self.W] = self.blindspot_f.copy()
        
    def close(self):
        self.cv2.destroyAllWindows()
        
    def taskDashboard(self):
        while(1):
            if(self.flag.schoolzone == True):
                self.addSchoolzone()
                if(self.flag.stopline == True):
                    self.addStopline()
                else:
                    self.subsStopline()
                if(self.flag.blindspot == True):
                    self.addBlindspot()
                else:
                    self.subsBlindspot()
            else:
                self.subsSchoolzone()
            self.show()
    def startDashboard(self):
        self.threadDashboard = self.Process(target=self.taskDashboard)
        self.threadDashboard.start()
            
        
class Flag:
    from threading import Thread
    import time
    def __init__(self):
        self.schoolzone = False
        self.stopline = False
        self.blindspot = False
        self.aeb = False
        self.lane_err = 0
        
    def printState(self):
        while(1):
            print(self.schoolzone, self.stopline, self.blindspot, self.aeb)
            self.time.sleep(2)

    def startPrintState(self):
        self.threadState = self.Thread(target=self.printState)
        self.threadState.start()
            
        
class ImageSet:
    import cv2
    import numpy as np
    import glob
    
    def __init__(self, usage, imgdir):
        self.usage = usage
        self.dir = imgdir
        path = imgdir + "/*.jpg"
        self.imageFname = self.glob.glob(path)
        self.amount = len(self.imageFname)
        self.index = 0
    def callImage(self):
        image = self.cv2.imread(self.imageFname[self.index],self.cv2.IMREAD_COLOR)
        if(self.index < (self.amount)):
            self.index = self.index + 1
        else: self.index = 0
        
        return image
        
        
        
