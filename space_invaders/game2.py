import math

import pygame
from math import sqrt
from pygame.locals import *
from space_invaders.player import Player
from space_invaders.enemy import Enemy
from space_invaders.bullet import Bullet
import numpy as np
import random


class Game:
    def __init__(self, screen, rows=3, cols=6, game_speed=1, enemies_attack=True, enemy_attackspeed=0.01, ai=False,
                 danger_threshold=175):
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.col_positions, self.row_positions = self.init_cols_rows()
        self.bg_color = (0, 0, 0)
        self.clock = pygame.time.Clock()
        self.FPS = game_speed * 60
        self.player_image = "../assets/images/player.png"
        self.enemy_image = "../assets/images/enemy.png"
        self.bullet_image = "../assets/images/bullet.png"
        self.player = Player(400, 500, self.player_image)
        self.enemies = pygame.sprite.Group()
        self.enemy_speed_x = 10
        self.enemy_speed_y = 0
        self.enemies_attack = enemies_attack
        self.enemy_attackspeed = enemy_attackspeed
        self.enemies_matrix = []
        self.bullets = pygame.sprite.Group()
        self.spawn_enemies(rows=rows, cols=cols)
        self.score = 0
        self.font = pygame.font.Font("../assets/fonts/game_font.otf", 36)
        self.lives_font = pygame.font.Font("../assets/fonts/game_font.otf", 24)
        self.game_over = False
        self.ai = ai
        self.danger_threshold = danger_threshold
        self.counter = 0

        self.fields = [False] * 13

    def init_cols_rows(self):
        spacing_x = 120
        spacing_y = 100
        beginning_offset_x = 80
        beginning_offset_y = 80
        col_positions = {}
        row_positions = {}
        for col in range(self.cols):
            col_positions[col] = beginning_offset_x + col * spacing_x
        for row in range(self.rows):
            row_positions[row] = beginning_offset_y + row * spacing_y
        return col_positions, row_positions

    def enemy_bullet_positions(self, player, bullets, distance_threshold):
        player_x, player_y = player.rect.center

        angle_ranges = [(i * 36, (i + 1) * 36) for i in range(5)]
        distance_ranges = [distance_threshold / 3, distance_threshold * 2 / 3, distance_threshold]

        self.fields = [False] * 13

        for bullet in bullets:
            if not bullet.player_bullet:
                bullet_x, bullet_y = bullet.rect.center

                gk = abs(player_y - bullet_y)
                ak = abs(player_x - bullet_x)
                total_distance = math.hypot(bullet_x - player_x, bullet_y - player_y)
                angle = math.degrees(math.atan(gk / ak))

                for i, (angle_min, angle_max) in enumerate(angle_ranges):
                    if angle_min <= angle <= angle_max:
                        for j, dist_max in enumerate(distance_ranges):
                            if j == 2 and i != 1 and i != 2 and i != 3:
                                continue
                            if total_distance <= dist_max:
                                self.fields[i + j * 5] = True
                                break

        self.draw_danger_area()
        return self.fields

    def get_state(self):
        player_x = self.player.rect.x
        player_y = self.player.rect.y
        state = []

        # add army pos
        state.append(self.enemies_matrix[0][0].rect.x - player_x)
        state.append(self.enemies_matrix[0][0].rect.y - player_y)
        #state.append(0)

        alive_in_column = [False] * self.cols

        for j in range(self.cols):
            for i in range(len(self.enemies_matrix)):
                enemy = self.enemies_matrix[i][j]
                if enemy is not None and enemy.alive():
                    alive_in_column[j] = True
                    break
        for alive in alive_in_column:
            state.append(alive)

        fields = self.enemy_bullet_positions(self.player, self.bullets, self.danger_threshold)
        for field in fields:
            state.append(field)

        # For old showcase:
        #state.append(False)
        #state.append(False)
        return state

    def reset(self):
        self.enemies_matrix = []
        self.enemies.empty()
        self.spawn_enemies(rows=self.rows, cols=self.cols)
        self.game_over = False
        self.bullets.empty()
        self.player.kill()
        self.player = Player(400, 500, self.player_image)
        self.score = 0

        return self.get_state()

    def spawn_enemies(self, rows, cols):
        for row in range(rows):
            enemy_row = []
            for col in range(cols):
                x = self.col_positions[col]
                y = self.row_positions[row]
                enemy = Enemy(x, y, self.enemy_image, col=col, row=row, col_pos=self.col_positions,
                              row_pos=self.row_positions)
                enemy_row.append(enemy)
                self.enemies.add(enemy)
            self.enemies_matrix.append(enemy_row)

    # def run(self):
    #     while not self.game_over:
    #         self.update()
    #         self.draw()
    #
    #         self.clock.tick(self.FPS)

    def handle_input(self, keys=None, action=None):
        # actions
        # 0 = noAction
        # 1 = left
        # 2 = right
        # 3 = leftShoot
        # 4 = rightShoot
        # 5 = shoot
        if self.ai is True:
            if action == 0:
                pass
            elif action == 1 or action == 2:
                self.player.update(action)
            elif action == 3 or action == 4:
                self.player.update(action)
                self.shoot()
            elif action == 5:
                self.shoot()
        else:
            if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                self.player.update(keys=keys)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    self.game_over = True
                if keys[pygame.K_SPACE]:
                    self.shoot()

    def is_allowed_to_shoot(self):
        does_player_bullet_exist = False
        for bullet in self.bullets:
            if bullet.player_bullet:
                does_player_bullet_exist = True
        return does_player_bullet_exist

    def shoot(self):
        does_player_bullet_exist = self.is_allowed_to_shoot()
        if not does_player_bullet_exist:
            bullet = Bullet(self.player.rect.x + self.player.rect.width / 2 - 2, self.player.rect.y,
                            self.bullet_image, -5, player_bullet=True)
            self.bullets.add(bullet)

    def remove_enemy(self, enemy):
        enemy.die()
        enemy.kill()
        self.enemies.remove(enemy)

    def any_enemies_alive(self):
        any_alive = False
        for enemy_list in self.enemies_matrix:
            for enemy in enemy_list:
                if enemy is not None:
                    if enemy.alive():
                        any_alive = True
        return any_alive

        # enemy.kill()
        # self.enemies.remove(enemy)
        # for enemy_list in self.enemies_matrix:
        #     if enemy in enemy_list:
        #         enemy_list.remove(enemy)
        #     if len(enemy_list) == 0:
        #         self.enemies_matrix.remove(enemy_list)


    def step(self, action=None):
        self.counter += 1
        if self.ai:
            self.handle_input(keys=None, action=action)
        else:
            keys = pygame.key.get_pressed()
            self.handle_input(keys=keys, action=None)

        old_score = self.score
        old_state = np.array(self.get_state())
        survival_reward = 1  # Define the survival reward

        enemy_moved_down = False

        for enemy in self.enemies:
            if enemy.rect.x < 10 and self.enemy_speed_x < 0 or enemy.rect.x > 790 - enemy.rect.width and self.enemy_speed_x > 0:
                self.enemy_speed_x = -self.enemy_speed_x
                self.enemy_speed_y = 10
                enemy_moved_down = True
                break
            if enemy.rect.bottom > self.screen.get_height():
                self.player.lives -= 1
                self.score -= 500
                if self.score < -1500:
                    self.score = -1500
                self.remove_enemy(enemy)
        if self.player.lives <= 0:
            self.game_over = True
        if self.counter % 30 == 0:
            self.enemies.update(self.enemy_speed_x, self.enemy_speed_y)
            self.enemy_speed_y = 0

        if self.enemies_attack:
            self.enemy_fire()

        self.bullets.update()

        for bullet in self.bullets:
            # Player bullet collision
            if bullet.player_bullet:
                enemy_hit = pygame.sprite.spritecollideany(bullet, self.enemies)
                if enemy_hit:
                    self.remove_enemy(enemy_hit)
                    bullet.kill()
                    self.score += 500
                elif bullet.rect.bottom <= 25:  # Check if the bullet has reached the top of the screen
                    self.score -= 100  # Apply a penalty for missing
            # Enemy bullet collision
            else:
                if pygame.sprite.collide_rect(bullet, self.player):
                    bullet.kill()
                    self.score -= 500
                    self.player.decrease_lives()
                    if self.player.lives <= 0:
                        self.player.kill()
                        self.game_over = True

        if pygame.sprite.spritecollideany(self.player, self.enemies) or not self.any_enemies_alive():
            self.score -= 1500
            if self.score < -1500:
                self.score = -1500
            self.game_over = True
        if self.game_over and self.ai is False:
            self.reset()

        done = self.game_over
        # Get the new state
        new_state = np.array(self.get_state())
        dodge_reward = self.calculate_dodge_reward(old_state, new_state, action)

        move_down_penalty = -100 if enemy_moved_down else 0

        return new_state, (self.score - old_score) + survival_reward + dodge_reward + move_down_penalty, done

    def calculate_dodge_reward(self, old_state, new_state, action):
        dodge_reward = 0
        # Determine if player was near a bullet in the old state
        for i, field in enumerate(old_state[-self.danger_threshold:]):
            if field:
                old_player_x, old_player_y = old_state[:2]  # Assuming these are the player's x and y coordinates
                new_player_x, new_player_y = new_state[:2]
                # Check if the action taken has increased the distance between the player and the bullet
                # This is a simple example and may need to be modified depending on how your state is structured
                if action == 1 and new_player_x < old_player_x:  # Action was move left
                    dodge_reward += 1
                elif action == 2 and new_player_x > old_player_x:  # Action was move right
                    dodge_reward += 1
                # Check if there is no bullet in the new state
                if not new_state[-self.danger_threshold:][i]:
                    dodge_reward += 1
        return dodge_reward


    def update(self, action=None):
        self.counter += 1
        if self.ai:
            self.handle_input(keys=None, action=action)
        else:
            keys = pygame.key.get_pressed()
            self.handle_input(keys=keys, action=None)

        old_score = self.score

        for enemy in self.enemies:
            if enemy.rect.x < 10 and self.enemy_speed_x < 0 or enemy.rect.x > 790 - enemy.rect.width and self.enemy_speed_x > 0:
                self.enemy_speed_x = -self.enemy_speed_x
                self.enemy_speed_y = 10
                break
            if enemy.rect.bottom > self.screen.get_height():
                self.player.lives -= 1
                self.score -= 500
                if self.score < -1500:
                    self.score = -1500
                self.remove_enemy(enemy)
        if self.player.lives <= 0:
            self.game_over = True
        if self.counter % 30 == 0:
            self.enemies.update(self.enemy_speed_x, self.enemy_speed_y)
            self.enemy_speed_y = 0

        if self.enemies_attack:
            self.enemy_fire()

        self.bullets.update()

        for bullet in self.bullets:
            # Player bullet collision
            if bullet.player_bullet:
                enemy_hit = pygame.sprite.spritecollideany(bullet, self.enemies)
                if enemy_hit:
                    self.remove_enemy(enemy_hit)

                    bullet.kill()
                    self.score += 100
            # Enemy bullet collision
            else:
                if pygame.sprite.collide_rect(bullet, self.player):
                    bullet.kill()
                    self.score -= 500
                    self.player.decrease_lives()
                    if self.player.lives <= 0:
                        self.player.kill()
                        self.game_over = True

        if pygame.sprite.spritecollideany(self.player, self.enemies) or not self.any_enemies_alive():
            self.score -= 1500
            if self.score < -1500:
                self.score = -1500
            self.game_over = True
        if self.game_over and self.ai == False:
            self.reset()

        return tuple(self.get_state()), self.score - old_score

    def enemy_fire(self):
        for i in range(len(self.enemies_matrix[0])):
            if random.random() < self.enemy_attackspeed:
                for j in range(len(self.enemies_matrix) - 1, -1, -1):
                    enemy = self.enemies_matrix[j][i]
                    if enemy is not None and enemy.alive():
                        bullet = Bullet(enemy.rect.x + enemy.rect.width // 2, enemy.rect.y + enemy.rect.height,
                                        self.bullet_image, 5, player_bullet=False)
                        self.bullets.add(bullet)
                        break


    def render(self, agent=None):

        self.screen.fill(self.bg_color)
        # Draw the score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 0))

        lives_text = self.lives_font.render(f"Lives: {self.player.lives}", True, (255, 255, 255))
        self.screen.blit(lives_text, (10, 40))

        self.screen.blit(self.player.image, self.player.rect)
        self.enemies.draw(self.screen)
        self.bullets.draw(self.screen)
        if agent is not None:
            best_action = agent.choose_action(self.get_state(), self)
            self.draw_best_action_arrow(best_action)

        # self.draw_danger_area()

        pygame.display.flip()

    def draw(self, agent=None):

        self.screen.fill(self.bg_color)
        # Draw the score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 0))

        lives_text = self.lives_font.render(f"Lives: {self.player.lives}", True, (255, 255, 255))
        self.screen.blit(lives_text, (10, 40))

        self.screen.blit(self.player.image, self.player.rect)
        self.enemies.draw(self.screen)
        self.bullets.draw(self.screen)
        if agent is not None:
            best_action = agent.choose_action(self.get_state(), self)
            self.draw_best_action_arrow(best_action)

        # self.draw_danger_area()

        pygame.display.flip()

    def draw_danger_area(self):
        player_x, player_y = self.player.rect.center
        danger_color = (255, 0, 0)
        pygame.draw.circle(self.screen, danger_color, (player_x, player_y), self.danger_threshold, 2)
        pygame.draw.circle(self.screen, danger_color, (player_x, player_y), self.danger_threshold / 3, 2)
        pygame.draw.circle(self.screen, danger_color, (player_x, player_y), (self.danger_threshold / 3) * 2, 2)

        start_point = (player_x, player_y)
        end_point0 = (start_point[0] + self.danger_threshold * math.cos(math.radians(0)),
                      start_point[1] - self.danger_threshold * math.sin(math.radians(0)))
        end_point1 = (start_point[0] + self.danger_threshold * math.cos(math.radians(36)),
                      start_point[1] - self.danger_threshold * math.sin(math.radians(36)))
        end_point2 = (start_point[0] + self.danger_threshold * math.cos(math.radians(72)),
                      start_point[1] - self.danger_threshold * math.sin(math.radians(72)))
        end_point3 = (start_point[0] + self.danger_threshold * math.cos(math.radians(108)),
                      start_point[1] - self.danger_threshold * math.sin(math.radians(108)))
        end_point4 = (start_point[0] + self.danger_threshold * math.cos(math.radians(144)),
                      start_point[1] - self.danger_threshold * math.sin(math.radians(144)))
        end_point5 = (start_point[0] + self.danger_threshold * math.cos(math.radians(180)),
                      start_point[1] - self.danger_threshold * math.sin(math.radians(180)))

        pygame.draw.line(self.screen, danger_color, start_point, end_point0)
        pygame.draw.line(self.screen, danger_color, start_point, end_point1)
        pygame.draw.line(self.screen, danger_color, start_point, end_point2)
        pygame.draw.line(self.screen, danger_color, start_point, end_point3)
        pygame.draw.line(self.screen, danger_color, start_point, end_point4)
        pygame.draw.line(self.screen, danger_color, start_point, end_point5)
        # player_x, player_y = self.player.rect.center
        # danger_color = (255, 0, 0)
        # pygame.draw.circle(self.screen, danger_color, (player_x, player_y), self.danger_threshold, 2)

    def draw_best_action_arrow(self, best_action):
        arrow_size = 20
        arrow_color = (255, 0, 0)

        if best_action == 1:  # Links
            pygame.draw.line(self.screen, arrow_color,
                             (self.player.rect.left, self.player.rect.centery),
                             (self.player.rect.left - arrow_size, self.player.rect.centery), 3)
        elif best_action == 2:  # Rechts
            pygame.draw.line(self.screen, arrow_color,
                             (self.player.rect.right, self.player.rect.centery),
                             (self.player.rect.right + arrow_size, self.player.rect.centery), 3)
        elif best_action == 3:  # Links schießen
            pygame.draw.line(self.screen, arrow_color,
                             (self.player.rect.left, self.player.rect.centery),
                             (self.player.rect.left - int(arrow_size * 1.5), self.player.rect.centery - arrow_size), 3)
        elif best_action == 4:  # Rechts schießen
            pygame.draw.line(self.screen, arrow_color,
                             (self.player.rect.right, self.player.rect.centery),
                             (self.player.rect.right + int(arrow_size * 1.5), self.player.rect.centery - arrow_size), 3)
