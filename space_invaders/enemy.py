import pygame

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path):
        super().__init__()
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self, speed_x, speed_y):
        self.rect.x += speed_x
        self.rect.y += speed_y
