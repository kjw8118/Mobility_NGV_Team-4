'''
Created on 2020. 2. 10.

@author: jinwon
'''
class A:
    def __init__(self):
        self.flagA = False
        self.flagB = False
        self.flagC = False
        


class Operator:
    def __init__(self, flag):
        self.flag = flag
        

Flag = A()

Ctrl = Operator(Flag)

print(Flag.flagA)

Ctrl.flag.flagA = True

print(Flag.flagA)

        
        
        
        