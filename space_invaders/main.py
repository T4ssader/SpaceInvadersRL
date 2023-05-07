import pygame
from space_invaders.game import Game


# from space_invaders.rl_agent.training import train_agent

def main():
    pygame.init()

    # Set up space_invaders display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Space Invaders")

    # Create a space_invaders instance
    game = Game(screen=screen)

    # Uncomment the following line to train the reinforcement learning agent
    # train_agent(game)

    game.run()





if __name__ == "__main__":

    x = 0.9
    for i in range(160000):
        x= x * 0.99999
    print(x)

    main()
