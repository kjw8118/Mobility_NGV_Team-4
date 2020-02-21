from schoolzone import*
import time

if __name__ == '__main__':
    print("Initializing")
    flag = Flag()
    Motor = Actuator(flag)
    X4 = Lidar(flag)
    Img = ImageProcessing([480, 640],flag)
    
    
    print("Now begin")
    X4.start()
    Img.start()
    time.sleep(2)
    Motor.start()
    
