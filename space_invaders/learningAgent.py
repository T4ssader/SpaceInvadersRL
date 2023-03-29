import numpy as np
import time
import random


class QLearningAgent:
    def __init__(self, actions, epsilon=0.3, gamma=0.99, alpha=0.5):
        self.q_table = {}
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.prev_state = None
        self.prev_action = None

    def get_q_value(self, state, action):
        state = tuple(state)  # convert state to a tuple
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state][action]

    def update(self, reward, state, action):
        if self.prev_state is not None:
            prev_q_value = self.get_q_value(self.prev_state, self.prev_action)
            max_q_value = max([self.get_q_value(state, a) for a in self.actions])
            new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
            self.q_table[tuple(self.prev_state)][self.prev_action] = new_q_value
        self.prev_state = state
        self.prev_action = action

    def choose_action(self, state):
        #print(self.epsilon)
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            #print(action)
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q_value = max(q_values)
            count_max = q_values.count(max_q_value)
            if count_max > 1:
                best_indices = [i for i in range(len(self.actions)) if q_values[i] == max_q_value]
                i = random.choice(best_indices)
            else:
                i = q_values.index(max_q_value)
            action = self.actions[i]
        return action

    def reset(self):
        self.prev_state = None
        self.prev_action = None

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma


import pygame
from space_invaders.game import Game

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Space Invaders")

    game = Game(screen, rows=5, cols=8, game_speed=0.5, enemies_attack=False, enemy_attackspeed=0.01, ai=True)
    agent = QLearningAgent(actions=[0, 1, 2, 3, 4], epsilon=0.3, gamma=0.99, alpha=0.5)
    #game.run()
    for i in range(1000):
        game.reset()
        agent.reset()
        state = game.get_state()
        action = agent.choose_action(state)
        score = 0
        while not game.game_over:
            next_state, reward = game.update(action)
            score += reward
            next_action = agent.choose_action(next_state)
            time.sleep(0.01)
            agent.update(reward, state, action)
            game.draw()
            state = next_state
            action = next_action
            if game.game_over:
                print(f"Episode {i}: score={score}")
        if i % 100 == 0:
            agent.set_epsilon(agent.epsilon * 0.95)

    pygame.quit()
