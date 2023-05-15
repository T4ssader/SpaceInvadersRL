import pygame

from CartPole import CartPole
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

def playCartPole():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    cartpole = CartPole(screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    cartpole.step(action=-1)
                elif event.key == pygame.K_RIGHT:
                    cartpole.step(action=1)
        screen.fill((0, 0, 0))
        cartpole.render()
        pygame.display.flip()
    pygame.quit()






if __name__ == "__main__":

    x = 0.9
    for i in range(100):
        x= x * 0.99
    print(x)

    #playCartPole()
    #main()
