'''
Created on 2020. 2. 10.

@author: jinwon
'''

class coral:
    from threading import Thread
    import time


    def __init__(self):
        self.itr = 0
        
    def start(self,t):
        self.t = t
        self.thread = self.Thread(target=self.run)
        self.thread.start()
    def printtxt(self,t):
        self.prt_t = t
        self.prtThread = self.Thread(target=self.prt)
        self.prtThread.start()
    def run(self):
       # while(1):
            self.time.sleep(self.t)
            self.itr = self.itr+1
    def prt(self):
        #while(1):
            self.time.sleep(self.prt_t)
            print(self.itr)
class wait:
    from threading import Thread
    import time


    def __init__(self):
        self.itr = 0
        
    def start(self,t):
        self.t = t
        self.thread = self.Thread(target=self.run)
        self.thread.start()
        
    def run(self):
        self.time.sleep(self.t)

import time
import os
from multiprocessing import Process







#A = coral()
#B = txtprint()

B =  wait()
C =  wait()
D =  wait()
E =  wait()

#while(A.itr <= 20):
#t1 = time.time()
#A.start(2)
#A.printtxt(0.5)
#print(A.thread.join(),A.prtThread.join())
t1 = time.time()
B.start(5)
C.start(3)
D.start(1)
E.start(2)
B.thread.join()
C.thread.join()
D.thread.join()
E.thread.join()
t2 = time.time()
print(t2-t1)
#B.thread.join()
#B.thread.start()
#    A.thread.join()
#B.thread.join()

#t2 = time.time()
#    time.sleep(5)
#    print((t2-t1),A.itr)#, B.itr)
    

#print(A.itr)
    
            
        
        
    