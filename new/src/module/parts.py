
class Actuator:
    import Adafruit_PCA9685
    from threading import Thread
    import time

    def __init__(self, channels, flag):
        self.flag = flag
        self.servo_channel = channels[0]
        self.dcmotor_channel = channels[1]
        print(self.dcmotor_channel, self.servo_channel)
        self.PCA9685 = self.Adafruit_PCA9685.PCA9685()
        #self.dcmotor = self.Adafruit_PCA9685.PCA9685()
         
    def getServoConfig(self, pwmLeft, pwmMid, pwmRight):
        self.servo_dir = [pwmLeft, pwmMid, pwmRight]
        self.servo_unit = abs(pwmRight - pwmMid)

    def getDCmotorConfig(self, pwmForward, pwmBrake, pwmRear):
        self.dcmotor_dir = [pwmForward, pwmBrake, pwmRear]
    
    def initDev(self):
        #print("InitDev Begin")
        self.PCA9685.set_pwm_freq(60)
        self.PCA9685.set_pwm(self.servo_channel, 0 , self.servo_dir[1])
        #print("InitDev End")
        
#        self.dcmotor.set_pwm_freq(60)
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
        
    def depart(self):
        print("Depart", self.dcmotor_dir[0])
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[0])
        
    def getPwmServo(self, direction):
        if(direction == -1): return self.servo_dir[0]
        elif(direction == 0): return self.servo_dir[1]
        elif(direction == 1): return self.servo_dir[2]
        else: return self.servo_dir[1]
        
    def getPwmDCmotor(self, direction):
        if(direction == -1): return self.dcmotor_dir[0]
        elif(direction == 0): return self.dcmotor_dir[1]
        elif(direction == 1): return self.dcmotor_dir[2]
        else: return self.dcmotor_dir[1]
        
    def drive(self, pwm):
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, pwm)
        
    def steering(self):#, direction):
        direction = (self.lane_err) * self.servo_unit + self.servo_dir[1]
        self.PCA9685.set_pwm(self.servo_channel, 0, direction)
        
    def AEB(self):
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[2])
        self.time.sleep(2)
        #self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[0])
        #self.time.sleep(0.1)
        self.PCA9685.set_pwm(self.dcmotor_channel, 0, self.dcmotor_dir[1])
        
        while(1): pass
            


#class Avoidance: 
#    import math
    
class X4:
    from multiprocessing import Process
    from threading import Thread
    from PyLidar3 import YdLidarX4 as pylidar
    import cv2
    import numpy as np
    import math
    
    def __init__(self, port, flag):
        self.flag = flag
        self.port = port
        self.freq = self.cv2.getTickFrequency()
        self.lidar = self.pylidar(self.port)
        
    def init(self):
        if(self.lidar.Connect()):
            print(self.lidar.GetDeviceInfo())
            self.gen = self.lidar.StartScanning()
            return 1
        else:
            print("Error connecting to X4")
            return 0

    def scan(self):
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
        
    def scanFiltered(self, th=50):
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
        self.x = self.np.full(360,0)
        self.y = self.np.full(360,0)
        for i in range(0,360):
            self.x[i] = self.distance[i] * self.math.cos(self.math.radians(i))
            self.y[i] = self.distance[i] * self.math.sin(self.math.radians(i))
    def getTOF(self):
        self.tof = ~(self.distance==0)*1
    
    def veiwMap(self, res, boundary):
        self.map = self.np.full(res,255,dtype='uint8')
        self.cv2.circle(self.map,(int(res[1]/2),int(res[0]/2)),1,(0,0),1)
        for i in range(0,360):
            if(self.distance[i] <= boundary):
                self.cv2.circle(self.map,(int(self.x[i]/(2*boundary)*res[1]+res[1]/2),int(self.y[i]/(2*boundary)*res[0]+res[0]/2)),1,(0,0),-1)
        self.cv2.imshow('veiwMap',self.map)
        
    def getAvoidance(self, th_w, th_h):
        self.detectLimit  = th_h
        self.angleUpper = int(270+self.math.degrees(self.math.atan(th_w/10)))
        self.angleLower = int(270+self.math.degrees(self.math.atan(th_w/th_h)))
        self.flagAvoidance = 0
        self.stateAvoidance = 0
        
    def avoidance(self):
        if(self.flag.schoolzone == True):
            if(self.flag.blinspot == False):
                if(self.np.sum(self.tof[self.angleUpper:(self.angleUpper+4)]) == 4 ):
                    if((self.detecLimit-10) <= (self.np.sum(self.distance[self.angleUpper:(self.angleUpper+4)])/4) <= (self.detecLimit)):
                        self.flag.blindspot = True
            else:
                if(self.stateAvoidance == 0):
                    if(self.np.sum(self.tof[self.angleUpper:(self.angleUpper+4)]) == 0):
                        self.stateAvoidance = 1
                        
                else:
                    if(self.np.sum(self.tof[(self.angleLower-4):self.angleLower]) == 0):
                        self.stateAvoidance = 0
                        self.flag.blindspot = False
                    
                
    def detectAEB(self):
        front = self.distance[240:300]
        if(self.np.sum((front < 100)*1) > 20):
            return 1
        else: return 0
        
    def taskScan(self):
        while(1):
            self.scanFiltered()
            
    def startScan(self):
        #self.init()
        self.threadScan = self.Process(target=self.taskScan)
        self.threadScan.start()
        self.startLidarDetect()
        
        
    def taskLidarDetect(self):
        while(1):
            if(self.flag.schoolzone == True):
                self.getTOF()
                self.getAvoidance(50,800)
                self.Avoidance()
            
    def startLidarDetect(self):
        self.threadLidarDetect = self.Process(target=self.taskLidarDetect)
        self.threadLidarDetect.start()
        
"""        
    def __init__(self,port):    
        self.Lidar = self.X4(port)
        self.Lidar.init()
        
    def detect(self, th_w, th_h):
        
        angle1 = self.math.degrees(self.math.atan(th_w/th_h))
        angle2 = self.math.degrees(self.math.atan(th_w/(th_h*0.25)))
        pass
"""    
    
        
        
        
#print("Hello World!")
        
                
    
