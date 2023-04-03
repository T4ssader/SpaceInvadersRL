import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path):
        super().__init__()
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 10
        self.lives = 3

    def decrease_lives(self):
        self.lives -= 1

    def update(self, action=None, keys=None):
        if keys is not None:
            if keys[pygame.K_LEFT]:
                self.rect.x -= self.speed
            if keys[pygame.K_RIGHT]:
                self.rect.x += self.speed

            # Keep the player within the screen
            if self.rect.x < 0:
                self.rect.x = 0
            if self.rect.x > 800 - self.rect.width:
                self.rect.x = 800 - self.rect.width
        else:
            if action == 1 or action == 3:
                self.rect.x -= self.speed
            elif action == 2 or action == 4:
                self.rect.x += self.speed
            if self.rect.x < 0:
                self.rect.x = 0
            if self.rect.x > 800 - self.rect.width:
                self.rect.x = 800 - self.rect.width