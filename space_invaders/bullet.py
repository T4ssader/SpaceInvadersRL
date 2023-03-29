import pygame

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path, speed, player_bullet=True):
        super().__init__()
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed
        self.player_bullet = player_bullet

    def update(self):
        self.rect.y += self.speed

        # Remove the bullet if it goes off-screen
        if self.rect.y < 0 or self.rect.y > 600:
            self.kill()
