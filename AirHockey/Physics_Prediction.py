from ArmRobot import ArmRobot
from Camera import Camera
import numpy as np

class Physics_Prediction():
    def __init__(self, dt = False):
        self.xp = 0
        self.yp = 0

    def __getitem__(self,puck_bbox,table_bbox, dt=0):
        x = (puck_bbox[0] - table_bbox[0]) / (table_bbox[2]-table_bbox[0])
        y = (puck_bbox[1] - table_bbox[1]) / (table_bbox[3]-table_bbox[1])
        dx = x - self.xp
        dy = y - self.yp
        
        xc=1

        if dx >-0.0001: #slow or wrong way
            self.xp = x
            self.yp = y
            return(0.5)

        while(xc>0):
            if dy >0: #moving upwards
                #1=dy*t + y
                #t =(1-y)/dy
                xc = dx*(1-y)/dy + x
                if xc > 0:
                    x=xc # reset x to the impact point
                    y=1
                    dy = -dy #same angle, opposit direction
                    #print("hit top at:", xc)

            elif dy<0:
                #0=dy*t+y
                #t=(-y/dy)
                xc = dx*(-y)/dy + x
                if xc > 0:
                    x=xc # reset x to the impact point
                    y=0
                    dy = -dy #same angle, opposit direction
                    #print("hit bottom at:", xc)
            
            else:
                break

        #0=dx*t+x
        #t=-x/dx
        yc = dy*(-x/dx) + y
        yc=np.clip(yc, 0, 1)

        self.xp = x
        self.yp = y
        return yc
