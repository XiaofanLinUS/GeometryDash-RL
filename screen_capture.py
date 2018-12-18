from Game import Game
from Dasher import Dasher


import time
import torch


def train_an_eipsode(model, opt):
        opt.zero_grad()        
        loss = model.conclude_loss()
        loss.backward()
        opt.step()

def play_the_game(): 
        game = Game()

        model = Dasher()
        model = model.cuda()
        optmizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # init move
        move = 0
        while(game.running):
                reward, pause, input_img = game.get_report(move)

                # feed image input to the network
                input_img_tensor = torch.tensor(input_img).float()
                input_img_tensor = input_img_tensor.unsqueeze(0)
                input_img_tensor = input_img_tensor.unsqueeze(0).cuda()

                # make move upon the image
                move = model.make_move(input_img_tensor)

                # save the reward for last move
                model.save_reward(reward)
                print(move)
                if pause:
                        train_an_eipsode(model, optmizer)
                        time.sleep(1)
                        game.reset()


                

play_the_game()



