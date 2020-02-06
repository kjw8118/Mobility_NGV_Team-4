import time


class X4:
    from PyLidar3 import YdLidarX4 as pylidar
    import cv2
    import numpy as np
    import math
    
    def __init__(self, port):

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
        
    def scanFiltered(self, th):
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
    def getCartesian(self):
        self.x = self.np.full(360,0)
        self.y = self.np.full(360,0)
        for i in range(0,360):
            self.x[i] = self.distance[i] * self.math.cos(self.math.radians(i))
            self.y[i] = self.distance[i] * self.math.sin(self.math.radians(i))
            
    def veiwMap(self, res, boundary):
        self.map = self.np.full(res,255,dtype='uint8')
        self.cv2.circle(self.map,(int(res[1]/2),int(res[0]/2)),1,(0,0),1)
        for i in range(0,360):
            if(self.distance[i] <= boundary):
                self.cv2.circle(self.map,(int(self.x[i]/(2*boundary)*res[1]+res[1]/2),int(self.y[i]/(2*boundary)*res[0]+res[0]/2)),1,(0,0),-1)
        self.cv2.imshow('veiwMap',self.map)
        
        




if __name__ == '__main__':
    import cv2
    port = "/dev/ttyUSB0"
    print("Generate Instance")
    Lidar = X4(port)
    now = time.time()
    print("Initialize X4")
    if(Lidar.init()):
        while((time.time()-now)<= 30):
            Lidar.scanFiltered(15)
            Lidar.getCartesian()
            Lidar.veiwMap([800,800], 3000)
            #print(Lidar.distance)
            if(cv2.waitKey(1) == 27): break
        Lidar.stop()
        print("Scan is end")
    else: print("Not Connected")


