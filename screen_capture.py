import numpy as np
import cv2
import time

import ctypes
import os
from PIL import Image

LibName = 'prtscn.so'
AbsLibPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + LibName
grab = ctypes.CDLL(AbsLibPath)


def grab_screen(x1,y1,x2,y2):
    w, h = x2-x1, y2-y1
    size = w * h
    objlength = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte*objlength)()

    grab.getScreen(x1,y1, w, h, result)
    return Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)

def screen_record(): 
        last_time = time.time()
        while(True):
                printscreen =  np.array(grab_screen(0,40,800,640))
                print('loop took {} seconds'.format(time.time()-last_time))
                last_time = time.time()
                cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break        

screen_record()



