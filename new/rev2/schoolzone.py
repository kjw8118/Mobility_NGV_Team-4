from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate



class ImageProcessing:

    
    import numpy as np
    import cv2
    import math
    from threading import Thread
    from multiprocessing import Process
    from threading import Timer
    import os

    def __init__(self, resolution, flag):
        
        self.height, self.width = resolution
        self.resolution = self.height, self.width
        
        self.index=0
        self.mode = 0
        self.flag = flag
        
        self.camera = self.cv2.VideoCapture(-1)
        self.camera.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
        
        self._setROI()
        
        self._setCoral()
        
        self._getDashboard()
        
        self._getIpmMat()
        
        self._getMaskBG()
        
        
                
    def _setROI(self):
        self.ROI_W = int(self.width*0.2)
        self.ROI_H = int(self.height*0.35)
        self.ROI_far_pos = 0.65 # Set along your vehicle velocity and frane rate | Late: far, Fast: near
        self.ROI_far_rngH = slice(int(self.height*self.ROI_far_pos - self.ROI_H), int(self.height*self.ROI_far_pos))
        self.ROI_near_rngH = slice(int(self.height - self.ROI_H), int(self.height))
        self.ROI_rngW = slice(int(self.width/2 - self.ROI_W/2),int(self.width/2 + self.ROI_W/2))
        self.ROI_lane_rngH = slice(int(self.height*0.7 - self.ROI_H*0.7), int(self.height*0.7))
        self.ROI_lane_rngW = slice(int(self.width/2 - self.ROI_W*0.5),int(self.width/2 + self.ROI_W*0.5))
        
    def _getIpmMat(self):
        
        # Your camera inner & external parameters
        alpha = (7-90)*self.np.pi/180
        beta = 0
        gamma = 0
        dist = 300
        focal = 500
        
        # Calculating rotational transformation matrix
        A1 = self.np.array([[1,0,-self.width/2],[0,1,-self.height/2],[0,0,0],[0,0,1]],dtype='f')
        RX = self.np.array([[1,0,0,0],[0,self.math.cos(alpha), -self.math.sin(alpha),0],[0,self.math.sin(alpha),self.math.cos(alpha),0],[0,0,0,1]],dtype='f')
        RY = self.np.array([[self.math.cos(beta),0,-self.math.sin(beta),0],[0,1,0,0],[self.math.sin(beta),0,self.math.cos(beta),0],[0,0,0,1]],dtype='f')
        RZ = self.np.array([[self.math.cos(gamma),-self.math.sin(gamma),0,0],[self.math.sin(gamma),self.math.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]],dtype='f')
        R = self.np.dot(RX,self.np.dot(RY,RZ))
        T = self.np.array([[1,0,0,0],[0,1,0,0],[0,0,1,dist],[0,0,0,1]],dtype='f')
        K = self.np.array([[focal,0,self.width/2,0],[0,focal,self.height/2,0],[0,0,1,0]],dtype='f')
        self.IpmMat = self.np.dot(K,self.np.dot(T,self.np.dot(R,A1)))
        
    
        
        
    def _getMaskBG(self):
        # Mask for blank area when images were ipm mapped 
        tmp_blank = self.np.full((self.height, self.width,3), 255, dtype='uint8')
        tmp_ipm = self.cv2.warpPerspective(tmp_blank, self.IpmMat, (self.width, self.height), flags=self.cv2.INTER_CUBIC|self.cv2.WARP_INVERSE_MAP)
        self.maskBG =  self.cv2.bitwise_not(tmp_ipm)
        
    def _getDashboard(self):
        self.board_size = 320
        size = self.board_size
        self.board = self.np.full((size*2, size*2,3), 255, dtype='uint8')
        tmp_stopline = self.cv2.imread('dashboard/stopline_red.jpg',self.cv2.IMREAD_COLOR)
        self.icon_stopline = self.cv2.resize(tmp_stopline, dsize=(size,size), interpolation=self.cv2.INTER_AREA)
        tmp_blindspot = self.cv2.imread('dashboard/blind_spot_red.jpg',self.cv2.IMREAD_COLOR)
        self.icon_blindspot = self.cv2.resize(tmp_blindspot, dsize=(size,size), interpolation=self.cv2.INTER_AREA)
        self.icon_blank = self.np.full((size,size*2,3),255,dtype='uint8')
        self.icon_schoolzone = self.np.full((size,size*2,3),0,dtype='uint8')
        self.icon_schoolzone[:,:,2] = 255
        self.icon_subs = self.np.full((size,size,3),0,dtype='uint8')
        self.icon_subs[:,:,2] = 255
        
               
        
        
    def processing(self):
        # Calibration parameters by experiments
        CamMat = self.np.array([[314.484, 0, 321.999],[0, 315.110, 259.722],[ 0, 0, 1]],dtype='f')
        DistMat = self.np.array([ -0.332015,    0.108453,    0.001100,    0.002183],dtype='f')
        
        # For inRange function in opencv
        # Modify value along your brightness condition
        lower_k = self.np.array([0,0,0])
        upper_k = self.np.array([180,255,100])
        lower_r1 = self.np.array([0,50,50])
        upper_r1 = self.np.array([30,255,255])
        lower_r2 = self.np.array([150,50,50])
        upper_r2 = self.np.array([180,255,255])
        
        
        ret, frame = self.camera.read(); del(ret) # Now take frame from camera
        calibration = self.cv2.undistort(frame, CamMat, DistMat, None, CamMat) # Calibration because of wide angle camera
        tmp_ipm1 = self.cv2.warpPerspective(calibration, self.IpmMat, (self.width,self.height), flags=self.cv2.INTER_CUBIC|self.cv2.WARP_INVERSE_MAP) # Geometrical transform image to Top view perspective
        tmp_ipm2 = self.cv2.add(tmp_ipm1, self.maskBG) # It just merges ipm image with white background
        ipm = self.cv2.bilateralFilter(tmp_ipm2,9,50,50) # Just Filter
        self.result = ipm.copy()
        hsv = self.cv2.cvtColor(ipm, self.cv2.COLOR_BGR2HSV)
        gray = self.cv2.cvtColor(ipm, self.cv2.COLOR_BGR2GRAY)
        
        #canny = self.cv2.Canny(gray, 100, 200, 3) # If you want to use canny edge algorithm, activate this line
        threshold_inv = self.cv2.adaptiveThreshold(gray, 255, self.cv2.ADAPTIVE_THRESH_MEAN_C, self.cv2.THRESH_BINARY, 21, 5)
        threshold = self.cv2.bitwise_not(threshold_inv)
        #mask_k = self.cv2.inRange(hsv, lower_k, upper_k)
        #mask_k = canny.copy()
        mask_k = threshold.copy()
        self.mask_k = mask_k[self.ROI_far_rngH, self.ROI_rngW]#[self.ROI_far_rngH, self.ROI_rngW]
        self.mask_lane = mask_k[self.ROI_lane_rngH,self.ROI_lane_rngW]
        
        # Now you can get red mask for schoolzone detecting
        mask_r1 = self.cv2.inRange(hsv, lower_r1, upper_r1)
        mask_r2 = self.cv2.inRange(hsv, lower_r2, upper_r2)
        mask_r = self.cv2.add(mask_r1, mask_r2)
        self.mask_r = mask_r[self.ROI_near_rngH, self.ROI_rngW]
        
        
    def detectingSchoolzone(self):
        # Just counting red dots
        if((self.np.sum(self.mask_r)/255) > ((self.ROI_H)*(self.ROI_W)*0.2)):
            self.flag.schoolzone = True
        else:
            self.flag.schoolzone = False
    
    def laneDetect(self):
        # By Jinwon, Lane detecting algorithm.
        # adaptive detecting method
        # It need to improve
        frame = self.cv2.flip(self.mask_lane.copy(),0)
        
        H,W = frame.shape[0:2]
        
        lane_base = self.np.array(range(0,W))
        
        
        lane = self.np.full((H,1), int(W/2), dtype='uint32')
        laneL = self.np.full((H,1), int(W/2), dtype='uint32')
        laneR = self.np.full((H,1), int(W/2), dtype='uint32')
        
        
        num0 = self.np.sum(frame[0,:] != False)
        if(num0 != 0): lane[0] = int(self.np.sum(lane_base*frame[0,:])/(255*num0))
        else: lane[0] = int(W/2)
        
        for j in range(1,H):
            rangeL = range(0, int(lane[j-1]))
            rangeR = range(int(lane[j-1]), int(W))
            numL = self.np.sum(frame[j,rangeL] !=  False)
            numR = self.np.sum(frame[j,rangeR] !=  False)
            if(numL == 0)|(numR == 0): lane[j] = lane[j-1]
            else:
                laneL[j] = self.np.sum(lane_base[rangeL]*frame[j,rangeL])/(255*numL)
                laneR[j] = self.np.sum(lane_base[rangeR]*frame[j,rangeR])/(255*numR)
                lane[j] = (laneR[j] + laneL[j])/2
            self.mask_lane[int(H - j),int(lane[j])] = 255
        
        self.flag.lane_err = ((self.np.mean(lane)*2/W) -1) # Return method is various. It just return mean value
        
        
    def _setCoral(self, modeldir="Model"): # By github.com/EdjeElectronics & coral.ai
        CWD_PATH =self.os.getcwd()
        MODEL_NAME = modeldir
        GRAPH_NAME = "edgetpu.tflite" #If you don't have coral, "detect.tflite"
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
        
        
        
    def detectingStopline(self): # By github.com/EdjeElectronics & coral.ai
        
        image = self.mask_k.copy()
        imH,imW = image.shape[0:2]
        
        # Get image & general
        coral_frame = self.cv2.cvtColor(image, self.cv2.COLOR_GRAY2RGB)
        frame_rgb = self.cv2.cvtColor(coral_frame, self.cv2.COLOR_BGR2RGB)
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
        
        # Threshold
        self.coral_min_conf_threshold = 0.90
        
        self.flag.stopline = False
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(self.coral_scores)):
            if ((self.coral_scores[i] > self.coral_min_conf_threshold) and (self.coral_scores[i] <= 1.0)):
                if(self.coral_labels[int(self.coral_classes[i])] == "stopline"): self.flag.stopline = True
                
                
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(self.coral_boxes[i][0] * imH)) + self.ROI_far_rngH.start)
                xmin = int(max(1,(self.coral_boxes[i][1] * imW)) + self.ROI_rngW.start)
                ymax = int(min(imH,(self.coral_boxes[i][2] * imH)) + self.ROI_far_rngH.start)
                xmax = int(min(imW,(self.coral_boxes[i][3] * imW)) + self.ROI_rngW.start)
                
                self.cv2.rectangle(self.result, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
    
                # Draw label
                object_name = self.coral_labels[int(self.coral_classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(self.coral_scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = self.cv2.getTextSize(label, self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                self.cv2.rectangle(self.result, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), self.cv2.FILLED) # Draw white box to put label text in
                self.cv2.putText(self.result, label, (xmin, label_ymin-7), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        



    




    # Thread   
    def task(self):
        while(1):
            t1 = self.cv2.getTickCount()
            self.processing()
            self.laneDetect()
            self.detectingSchoolzone()
            self.flag.schoolzone = False #tmp
            
            
            if(self.mode == 1):
                self.detectingStopline()
            
            
            self.board[160:640,:,:] = self.result.copy()
            self.board[0:self.board_size,:,:] = self.icon_blank.copy()


            # Now, Mode Selector
            if(self.mode == 0):
                if(self.flag.schoolzone == True):
                    self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                    self.mode = 1
       
            elif(self.mode == 1):
                self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                if(self.flag.stopline == True):
                    self.board[0:self.board_size,0:self.board_size,:] = self.icon_stopline.copy()
                    self.mode = 2
                    self.flag.stop = True
                    
                                                                                
            elif(self.mode == 2):
                self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                self.board[0:self.board_size,0:self.board_size,:] = self.icon_stopline.copy()
                
                if(self.flag.depart == True):
                    self.flag.depart = False
                    self.flag.powerHandle = True
                    self.mode = 3
                    
            elif(self.mode == 3):
                self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                
                if(self.flag.refresh == True):
                    self.flag.refresh = False
                    self.flag.lidar = True
                    self.mode = 4
                    
            elif(self.mode == 4):
                self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                if(self.flag.blindspot == True):
                    self.mode = 5
                    self.flag.slow = True
                    
            elif(self.mode == 5):
                self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                self.board[0:self.board_size,self.board_size:(self.board_size*2),:] = self.icon_blindspot.copy()
                if(self.flag.blindspot == False):
                    self.mode = 6
                    self.flag.slow = False
                    
            elif(self.mode == 6):
                self.board[0:self.board_size,:,:] = self.icon_schoolzone.copy()
                if(self.flag.schoolzone == False):
                    self.board[0:self.board_size,:,:] = self.icon_blank.copy()
                    self.mode = 7
                    
               
            self.cv2.imshow('Dashboard', self.board)
            t2 = self.cv2.getTickCount()
            freq = self.cv2.getTickFrequency()
            #print(freq/(t2-t1))

            self.cv2.waitKey(1)
            
            if(self.mode == 7):
                self.flag.end = True
                break
        self.cv2.destroyAllWindows()

        
    def start(self):
        self.thread = self.Thread(target=self.task)
        self.thread.start()
    def startLane(self):
        self.threadLane = self.Thread(target=self.taskLane)
        self.threadLane.start()

        
        
        
        
        
        
    
class Actuator:
    import Adafruit_PCA9685
    from threading import Thread
    from threading import Timer
    import time

    def __init__(self, flag):
    
        self.flag = flag
        

        
        self.servo_channel = 0
        self.dcmotor_channel = 1
       
        self.PCA9685 = self.Adafruit_PCA9685.PCA9685()

        # Take value by calibration experitments
        self.servo_dir = [334, 374, 430]
        self.servo_unit = 40
        
        self.dcmotor_dir = [406, 390, 365]
        
        self.cnt = 0
        
        self._setDev()
        
    def _setDev(self):
        self.PCA9685.set_pwm_freq(60)
        self.PCA9685.set_pwm(self.servo_channel, 0 , self.servo_dir[1])        
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
        print("Ready")
        self.time.sleep(3)
    def depart(self):
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[0]+8)
        self.time.sleep(0.6) # For initial motor control

    def steering(self):
        if(self.flag.powerHandle == False):
            direction = int((self.flag.lane_err)*1.5 * self.servo_unit + self.servo_dir[1])
        else:
            direction = int((self.flag.lane_err)*2.5 * self.servo_unit + self.servo_dir[1])
        self.PCA9685.set_pwm(self.servo_channel, 0, direction)
 
    def task(self):
        if(self.flag.stop == True):
            self.flag.stop = False
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
            self.time.sleep(0.3)
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[2]-10)
            self.time.sleep(1.5)
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
            self.time.sleep(3)
            self.flag.depart = True
            self.depart()
            
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[0])
            self.time.sleep(2)
            
            self.flag.refresh = True
            
        elif(self.flag.slow == True):
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[0]-3)
            self.steering()
            
                
        elif(self.flag.end == True):
            self.flag.end = False
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
            self.time.sleep(1)
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[2])
            self.time.sleep(0.5)
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
            self.time.sleep(2)
            self.timer.cancel()
            return#break
            
        else:
            self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[0])
            self.steering()
        
        # Actuator module is runned in Timer thread
        self.timer = self.Timer(0.05,self.task)
        self.timer.start()
        
            
            
            
    def start(self):
        self.depart()
        self.task()
    def stop(self):
        self.PCA9685.set_pwm(self.dcmotor_channel,0,self.dcmotor_dir[1])
    



        
class Lidar: # By PyLidar3
    from multiprocessing import Process
    from threading import Thread
    from threading import Timer
    from PyLidar3 import YdLidarX4 as pylidar
    import cv2
    import numpy as np
    import math
    import time
    
    def __init__(self, flag):
        self.flag = flag
        self.port = "/dev/ttyUSB0"
        self.freq = self.cv2.getTickFrequency()
        
        while(1): # Infinite trial for connecting Lidar X4 
            self.lidar = self.pylidar(self.port)
            self.time.sleep(1)
            if(self.lidar.Connect()):
                print(self.lidar.GetDeviceInfo())
                self.gen = self.lidar.StartScanning()
                break
            else:
                print("Error connecting to X4")
            self.time.sleep(2)   

    def scan(self): # Just scan in PyLidar Example
        t1 = self.cv2.getTickCount()
        data_dict = next(self.gen)
        arr = self.np.full(len(data_dict),None)
        for i in range(0,len(data_dict)):
            arr[i] = data_dict[i]
        self.distance = arr
        t2 = self.cv2.getTickCount()
        
        self.fps = self.freq/(t2-t1)

        
    def stop(self):
        self.lidar.StopScanning()
        self.lidar.Disconnect()
        
    def scanFiltered(self, th=50): # Scan data with filter made by Jinwon Kim
        t1 = self.cv2.getTickCount()
        data_dict = next(self.gen)
        self.distance = self.np.full(len(data_dict),None)
        self.distance[0] = data_dict[0]
        for i in range(1,len(data_dict)):
            if(abs(data_dict[i] - data_dict[i-1]) > th):
                self.distance[i] = self.distance[i-1]
            else:
                self.distance[i] = data_dict[i]
        t2 = self.cv2.getTickCount()
        self.fps = self.freq/(t2-t1)
        
    def getPOV(self,angle1,angle2,mode=0):
        if(angle1>=angle2):
            print("Angle 1 must be smaller than Angle 2 !!!")
            self.pov = None
            return
        
        if(mode == 1):
            self.pov = self.tof[angle1:angle2]
        else: self.pov = self.distance[angle1:angle2]
        
    def getCartesian(self):
        x = self.np.full(360,0)
        y = self.np.full(360,0)
        for i in range(0,360):
            x[i] = self.distance[i] * self.math.cos(self.math.radians(i))
            y[i] = self.distance[i] * self.math.sin(self.math.radians(i))
        self.x = x
        self.y = y
        
    def getTOF(self):
        self.tof = ~(self.distance==0)*1
    
    def getMap(self, boundary, res=()):
        self.map = self.np.full(res,255,dtype='uint8')
        self.cv2.circle(self.map,(int(res[1]/2),int(res[0]/2)),1,(0,0),1)
        for i in range(0,360):
            if(self.distance[i] <= boundary):
                self.cv2.circle(self.map,(int(self.x[i]/(2*boundary)*res[1]+res[1]/2),int(self.y[i]/(2*boundary)*res[0]+res[0]/2)),1,(0,0),-1)
        self.cv2.imshow('veiwMap',self.map)
        
    
    def detect(self, Xbound, Ybound): # Obstacle algorithm by Jinwon Kim
        lower_x, upper_x = Xbound
        upper_y_tmp, lower_y_tmp  = Ybound
        upper_y = -upper_y_tmp
        lower_y = -lower_y_tmp
        det = ((lower_x<self.x)&(self.x<upper_x))*((lower_y<self.y)&(self.y<upper_y))
        num_det = self.np.sum(det)
        print(num_det)
        if(num_det > 10): self.flag.blindspot = True
        else: self.flag.blindspot = False

        
    def task(self):
        while(1):
            t1 = self.cv2.getTickCount()
            self.scanFiltered(50)
            self.getCartesian()
            #elf.detect((300,700),(100,1000))
            if(self.mode == 0):
                if(self.flag.lidar == True):
                    self.mode = 1
                    
            elif(self.mode == 1):
                if(self.blindspot == True):
                    self.flag.blindspot = True
            t2 = self.cv2.getTickCount()
            freq = self.cv2.getTickFrequency()
            #print(freq/(t2-t1))
                
    def start(self):
        self.thread = self.Process(target=self.task)
        self.thread.start()
                    
        
        
        
class Flag: # Data structure for class
    def __init__(self):
        self.schoolzone  = False
        self.stopline = False
        self.blindspot = False
        self.stop = False
        self.depart = False
        self.refresh = False
        self.lidar = False
        self.slow = False
        self.end = False
        self.powerHandle = False
        self.lane_err = 0
