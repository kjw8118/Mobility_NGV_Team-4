'''
Created on 2020. 2. 9.

@author: jinwon
'''

from module import general
from module import parts
import cv2
import time

pwmLeft = 330 #Max @ 310
pwmMid = 374
pwmRight = 410 #Max @ 450

pwmForward = 406 #Max @ 420
pwmBrake = 390
pwmRear = 365 #Max @ 365


if __name__ == '__main__':
    
    # Define Object
    Flag = general.Flag()
    #Lidar = parts.X4("/dev/ttyUSB0",Flag)
    #Actuator =  parts.Actuator([0,1],Flag)
    #Coral = parts.SSD("Model",Flag)
    Img = general.ImgProc([480, 640],Flag)
    Dashboard = general.Dashboard(Flag)
    
    # Setup Device
    #Lidar.init()
    #Lidar.getAvoidance(50,600)
    #Actuator.getServoConfig(pwmLeft, pwmMid, pwmRight)
    #Actuator.getDCmotorConfig(pwmForward, pwmBrake, pwmRear)
    #print(Actuator.dcmotor_dir, Actuator.servo_dir)
    #Actuator.initDev()
    Img.getMaskBG()
    #freq = cv2.getTickFrequency()
    
    
    # Begin
    #Actuator.depart()#pwmForward)
    #Actuator.PCA9685.set_pwm = (
    #Lidar.scanFiltered()
    Dashboard.show()
    print("Depart")
    start_time = time.time()
    while(1):#(time.time()- start_time) <= 4):#not Lidar.detectAEB()):
        t1 = cv2.getTickCount()
        Img.getFrame()
        Img.getCalibration()
        Img.getIPM()
        Img.getMask()
        Img.detectSchoolzone()
        
        if(Flag.schoolzone == True):
            Dashboard.addSchoolzone()
            #Lidar.scanFiltered()
            Img.inference()#Img.mask_k)
            Img.detect(0.98)
            if( Flag.stopline == True):
                #pass
                Dashboard.addStopline()
            else:
                #pass
                Dashboard.subsStopline()
            #Lidar.avoidance()
            #if(Lidar.stateAvoidance == 1):
            #    Dashboard.addBlindspot()
            #else:
            #    Dashboard.subsBlindspot()
                
        else:
            pass
            Dashboard.subsSchoolzone()
            #Dashboard.subsStopline()
            #Dashboard.subsBlindspot()
            
        t2 = cv2.getTickCount()
        #print("in Loop")

        #Actuator.depart()#pwmForward)
        Dashboard.show()
        #print(freq/(t2-t1))
        time.sleep(0.5)
            
            
        
    #Actuator.drive(pwmBrake)    
    #Actuator.AEB()
    
    #Lidar.stop()
    
    
    
    
    
