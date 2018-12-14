import numpy as np
import cv2
import time

import ctypes
import os
from PIL import Image

from pynput.keyboard import Key, Controller

GAMOVER = 'res/gameover_sign.png'
gameover_sign = cv2.imread(GAMOVER, cv2.IMREAD_GRAYSCALE)

pause = False

keyboard = Controller()
canny_threshold = 200
LibName = 'prtscn.so'
AbsLibPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + LibName
grab = ctypes.CDLL(AbsLibPath)


def checkGameOver_(grey_src_img):
        global pause
        if pause is not True:
                grey_src_img = grey_src_img[455:530, 520:600]
                result = cv2.matchTemplate(grey_src_img, gameover_sign, cv2.TM_CCOEFF_NORMED)
                threshod = 0.9
                location = np.where( result > threshod)
                pause = ((len(location[0]) != 0))


def checkGameOver(screen_diff):
        return np.sum(np.abs(screen_diff)) < 200


def grab_screen(x1,y1,x2,y2):
    w, h = x2-x1, y2-y1
    size = w * h
    objlength = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte*objlength)()

    grab.getScreen(x1,y1, w, h, result)
    result = np.frombuffer(result, dtype='uint8')
    result = result.reshape((h, w, 3))
    return result

def train(model):
        pass

def screen_record(): 
        global pause
        last_time = time.time()
        cv2.namedWindow('rl')
        cv2.moveWindow('rl', 1000, 0)
        prev_screen = np.zeros((600, 800), dtype=np.uint8)
        while(True):
                printscreen = grab_screen(0,64,800,600+64)
                printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
                

                screen_diff = printscreen - prev_screen
                prev_screen = printscreen
                #printscreen = cv2.Canny(printscreen, canny_threshold, canny_threshold * 1.2,  L2gradient=True)
                
                print('loop took {} seconds'.format(time.time()-last_time))

                checkGameOver_(printscreen)
                print(pause)
                if pause == False:
                        time.sleep(1)
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                
                cv2.imshow('rl', printscreen)
                last_time = time.time()
                print(printscreen.shape)
                
                if cv2.waitKey(25) & 0xFF == ord('c'):
                        pause = False

                if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break        

screen_record()



