from Game import Game
import time 

def screen_record(): 
        game = Game()
        while(game.running):
                reward, pause = game.get_report()
                if pause:
                        time.sleep(5)
                        game.reset()


                

screen_record()



