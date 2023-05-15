import pygame
import math
import numpy as np


class CartPole:
    def __init__(self, screen, cart_velocity=0, pole_angular_velocity=0, pole_angle=0, cart_position=0):
        self.screen = screen
        self.cart_velocity = cart_velocity
        self.pole_angular_velocity = pole_angular_velocity
        self.pole_angle = pole_angle
        self.cart_position = cart_position
        self.game_over = False
        self.score = 0
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 0.5  # half the pole's length
        self.polemass_length = (self.mass_pole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

    def reset(self):
        self.cart_velocity = 0
        self.pole_angular_velocity = 0
        self.pole_angle = 0
        self.cart_position = 0
        self.score = 0
        self.game_over = False
        return self.get_state()

    def step(self, action):
        force = self.force_mag if action == 1 else -self.force_mag
        temp = (force + self.polemass_length * self.pole_angle * self.pole_angular_velocity ** 2) / self.total_mass
        angular_acc = (self.gravity * math.sin(self.pole_angle) - math.cos(self.pole_angle) * temp) / (
                    self.length * (4.0 / 3.0 - self.mass_pole * math.cos(self.pole_angle) ** 2 / self.total_mass))
        acc = temp - self.polemass_length * angular_acc / self.total_mass

        self.cart_position += self.tau * self.cart_velocity
        self.cart_velocity += self.tau * acc
        self.pole_angle += self.tau * self.pole_angular_velocity
        self.pole_angular_velocity += self.tau * angular_acc

        if self.cart_position > 2.4 or self.cart_position < -2.4 or self.pole_angle > 1 or self.pole_angle < -1:
            self.game_over = True
            reward = -100
        else:
            if -0.15 < self.pole_angle < 0.15:
                reward = 10
            else:
                reward = 1
                # self.score += 1

        return self.get_state(), reward, self.game_over

    def get_state(self):
        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity])

    def render(self):
        self.screen.fill((0, 0, 0))
        # Screen parameters
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Scale factor for mapping world coordinates to screen coordinates
        world_width = 2.4 * 2  # consider 2.4 units on each side of the cart
        scale = screen_width / world_width

        # Map the cart's position to screen coordinates
        cart_x = int(
            self.cart_position * scale + screen_width / 2)  # add screen_width/2 to shift the origin to the screen's center
        cart_y = int(screen_height * 0.8)  # place the cart at 80% of the screen height

        # Calculate the pole's position in world coordinates
        pole_x = self.cart_position + math.sin(self.pole_angle) * self.length
        pole_y = math.cos(self.pole_angle) * self.length

        # Map the pole's position to screen coordinates
        pole_screen_x = int(pole_x * scale + screen_width / 2)
        pole_screen_y = cart_y - int(pole_y * scale)  # subtract from cart_y to flip the y-coordinate

        # Draw the cart and the pole
        pygame.draw.line(self.screen, (255, 255, 255), (cart_x, cart_y), (pole_screen_x, pole_screen_y), 5)
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(cart_x - 25, cart_y - 10, 50, 20))

        # Update the display
        pygame.display.flip()
