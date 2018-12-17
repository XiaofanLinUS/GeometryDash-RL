import ctypes
import os

import cv2
import time
import numpy as np

from pynput.keyboard import Key, Controller

GAMOVER_IMG = 'res/gameover_sign.png'

keyboard = Controller()

# For screen capture
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
    result = np.frombuffer(result, dtype='uint8')
    result = result.reshape((h, w, 3))
    return result


class Game():
    _gameover_sign = cv2.imread(GAMOVER_IMG, cv2.IMREAD_GRAYSCALE)
    def __init__(self):
        self.running = True
        self.best_time = 0
        self.begin_time = time.time()
        self.current_time = time.time()
        self.pause = False
        self.prev_screen = None
        cv2.namedWindow('game')
        cv2.moveWindow('game', 1000, 0)

    def get_report(self):
        current_time = time.time()
        time_gap = current_time - self.current_time
        reward = 0
        self.current_time = current_time
        current_screen = grab_screen(0,64,800,600+64)
        current_screen = cv2.cvtColor(current_screen, cv2.COLOR_BGR2GRAY)
        if self.prev_screen is None:
            screen_diff = np.zeros((600, 800), dtype=np.uint8)
        else:
            screen_diff = current_screen - self.prev_screen
        self.prev_screen = current_screen
        self.check_game_over(current_screen)
        if self.pause == False:
            keyboard.press(Key.up)
            keyboard.release(Key.up)
        else:
            time_alive = self.current_time - self.begin_time
            if time_alive > self.best_time:
                self.best_time= time_alive
                reward = 1
            else:
                reward = -1

        cv2.imshow('game', current_screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.running = False
        print(f'{time_gap} seconds since last report')
        print(f'reward: {reward}')

        return reward, self.check_game_over(current_screen)
        

    def reset(self):
        # signal the game to restart
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        self.begin_time = time.time()
        self.current_time = time.time()
        self.pause = False
        self.prev_screen = None
        print(f'Best time: {self.best_time}')

    def check_game_over(self, grey_src_img):
        if self.pause == True:
            return self.pause
        grey_src_img = grey_src_img[455:530, 520:600]
        
        result = cv2.matchTemplate(grey_src_img, type(self)._gameover_sign, cv2.TM_CCOEFF_NORMED)
        threshod = 0.9
        location = np.where( result > threshod)
        self.pause = ((len(location[0]) != 0))