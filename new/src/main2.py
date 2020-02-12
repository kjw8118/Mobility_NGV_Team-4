'''
Created on 2020. 2. 10.

@author: jinwon
'''
from module import general
from module import parts
import time

pwmLeft = 330 #Max @ 310
pwmMid = 370
pwmRight = 410 #Max @ 450

pwmForward = 406 #Max @ 420
pwmBrake = 390
pwmRear = 365 #Max @ 365


port = "/dev/ttyUSB0"

if __name__ == '__main__':
    Imageset = general.ImageSet(True,"images")
    Flag = general.Flag()
    ImgProc = general.ImgProc([240,320],Flag, Imageset)
    #Lidar = parts.X4(port,Flag)
    #Actuator = parts.Actuator([0,1], Flag)
    #Dashboard = general.Dashboard(Flag)
    
    ImgProc.getMaskBG()
    #ImgProc.initCoral()
    #Lidar.init()
    #Actuator.initDev()
    print("Initializing is done")
    time.sleep(1)
    #Actuator.depart()
    time.sleep(0.2)
    
    ImgProc.startPreproc()
    #ImgProc.startObjectDetect()
    #Lidar.startScan()
    #Lidar.startLidarDetect()
    #Dashboard.startDashboard()
    Flag.startPrintState()
    
    
    
