import pygame
from pygame import MOUSEBUTTONDOWN, KEYDOWN, K_UP, K_DOWN


class Menu:
    def __init__(self, screen, x, y, width, height):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bg_color = (50, 50, 50)
        self.font = pygame.font.Font("../assets/fonts/game_font.otf", 24)
        self.options = {
            "Epsilon": 0.3,
            "Gamma": 0.99,
            "Alpha": 0.5
        }
        self.active_option = None
        self.step_sizes = {
            "Epsilon": 0.1,
            "Gamma": 0.01,
            "Alpha": 0.1
        }

    def draw(self):
        pygame.draw.rect(self.screen, self.bg_color, (self.x, self.y, self.width, self.height))
        for index, (key, value) in enumerate(self.options.items()):
            text = self.font.render(f"{key}: {value}", True, (255, 255, 255))
            self.screen.blit(text, (self.x + 15, self.y + 15 + index * 40))

    def handle_input(self, event, agent):

        if event.type == MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if self.x < mouse_x < self.x + self.width and self.y < mouse_y < self.y + self.height:
                for index, (key, value) in enumerate(self.options.items()):
                    if self.y + 20 + index * 40 < mouse_y < self.y + 20 + (index + 1) * 40:
                        self.active_option = key
        elif self.active_option is not None and event.type == KEYDOWN:
            step_size = self.step_sizes[self.active_option]
            if event.key == K_UP:
                self.options[self.active_option] += step_size
            elif event.key == K_DOWN:
                self.options[self.active_option] -= step_size
            self.options[self.active_option] = round(self.options[self.active_option], 3)

        agent.set_epsilon(self.options["Epsilon"])
        agent.set_gamma(self.options["Gamma"])
        agent.set_alpha(self.options["Alpha"])

    def get_option(self, option):
        return self.options[option]

    def set_option(self, option, value):
        self.options[option] = value

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    menu = Menu(screen, 0, 0, 200, 600)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            menu.handle_input(event)
        screen.fill((0, 0, 0))
        menu.draw()
        pygame.display.flip()
        clock.tick(60)