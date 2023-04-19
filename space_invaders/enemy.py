import pygame


class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path, col=0, row=0, col_pos=None, row_pos=None):
        super().__init__()
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.col = col
        self.row = row
        self.col_pos = col_pos
        self.row_pos = row_pos
        self.shot = False

    def update(self, speed_x, speed_y):
        self.rect.x += speed_x
        self.rect.y += speed_y

    def die(self):
        self.shot = True

    def has_been_shot(self):
        return self.shot


    # def update(self, pos_x, pos_y):
    #     self.rect.x = pos_x
    #     self.rect.y = pos_y

    # def move(self):
    #     next_col = (self.col + 1) % (len(self.col_pos)-1)
    #     self.col = next_col
    #     if next_col == 0:
    #         next_row = (self.row + 1) % (len(self.row_pos)-1)
    #         self.row = next_row
    #     new_pos_x = self.col_pos[self.col]
    #     new_pos_y = self.row_pos[self.row]
    #     self.update(new_pos_x, new_pos_y)
