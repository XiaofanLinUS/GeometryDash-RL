from Game import Game
from Dasher import Dasher
import time 
import torch

def screen_record(): 
        game = Game()
        model = Dasher()
        model = model.cuda()
        move = 0
        while(game.running):
                reward, pause, input_img = game.get_report(move)
                input_img_tensor = torch.tensor(input_img).float()
                input_img_tensor = input_img_tensor.unsqueeze(0)
                input_img_tensor = input_img_tensor.unsqueeze(0).cuda()
                move = model.makeMove(input_img_tensor)
                print(move)
                if pause:
                        time.sleep(10)
                        game.reset()


                

screen_record()



