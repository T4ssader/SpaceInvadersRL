import pygame
import random
import numpy as np
import setuptools

from space_invaders.game import Game


# from space_invaders.rl_agent.training import train_agent
class Agent:
    # Konstanten
    epsilon = 0.1


    def __init__(self):
        pygame.init()
        # Set up space_invaders display
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Space Invaders")

        # Create a space_invaders instance
        self.game = Game(screen=screen, ai=True)

        # Uncomment the following line to train the reinforcement learning agent
        # train_agent(game)

        # game.run()
        self.game.reset()
        while not self.game.game_over:
            action = random.randint(0, 5)
            self.game.update(action)
            print(action)

            self.game.draw()
            self.game.clock.tick(self.game.FPS)

    def init_q_table(self):
        # [state,action] -> quality
        enemies = len(self.game.enemies_matrix)
        q_values = self.game.get_state()
        possible_enemy_position
        states = 800*600*
        for state
        [state,0]
        pass


agent = Agent()
