import numpy as np
import time
import random
from space_invaders.menu import Menu


class QLearningAgent:
    def __init__(self, actions, epsilon=0.3, gamma=0.99, alpha=0.5):
        self.q_table = {}  # Q-Tabelle zur Speicherung der Q-Werte
        self.actions = actions  # mögliche Aktionen des Agenten
        self.epsilon = epsilon  # Epsilon für die Epsilon-greedy Strategie
        self.gamma = gamma  # Discount-Faktor
        self.alpha = alpha  # Lernrate
        self.prev_state = None  # vorheriger Zustand des Agenten
        self.prev_action = None  # vorherige Aktion des Agenten

    def get_q_value(self, state, action):
        state = tuple(state)  # Konvertiere Zustand in ein Tuple
        if state not in self.q_table:  # Wenn der Zustand noch nicht in der Q-Tabelle ist, füge ihn hinzu
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state][action]

    def update(self, reward, state, action):
        if self.prev_state is not None:  # Wenn es einen vorherigen Zustand gibt
            # Berechne den neuen Q-Wert und aktualisiere die Q-Tabelle
            prev_q_value = self.get_q_value(self.prev_state, self.prev_action)
            max_q_value = max([self.get_q_value(state, a) for a in self.actions])
            new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
            self.q_table[tuple(self.prev_state)][self.prev_action] = new_q_value
        self.prev_state = state
        self.prev_action = action

    # Funktion zum Auswählen einer Aktion basierend auf der Epsilon-greedy Strategie
    def choose_action(self, state):
        # print(self.epsilon)
        if random.random() < self.epsilon:  # Wähle mit einer Wahrscheinlichkeit von Epsilon eine zufällige Aktion
            action = random.choice(self.actions)
            # print(action)
        else:  # Wähle sonst die Aktion mit dem höchsten Q-Wert
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q_value = max(q_values)
            count_max = q_values.count(max_q_value)
            if count_max > 1:  # Bei mehreren maximalen Q-Werten, wähle zufällig einen
                best_indices = [i for i in range(len(self.actions)) if q_values[i] == max_q_value]
                i = random.choice(best_indices)
            else:
                i = q_values.index(max_q_value)
            action = self.actions[i]
        return action

    # Funktion zum Zurücksetzen des Agenten
    def reset(self):
        self.prev_state = None
        self.prev_action = None

    # Funktionen zum Ändern von Epsilon, Alpha und Gamma
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

    game = Game(screen, rows=5, cols=11, game_speed=0.5, enemies_attack=True, enemy_attackspeed=0.001, ai=True)
    agent = QLearningAgent(actions=[0, 1, 2, 3, 4], epsilon=0.3, gamma=0.99, alpha=0.7)
    # game.run()
    game.menu.set_option("Epsilon", agent.epsilon)
    game.menu.set_option("Alpha", agent.alpha)
    game.menu.set_option("Gamma", agent.gamma)
    for i in range(1000):
        game.reset()
        game.game_over = False
        agent.reset()
        state = game.get_state()
        action = agent.choose_action(state)
        score = 0
        while not game.game_over:
            # print((agent.epsilon, agent.alpha, agent.gamma))
            next_state, reward = game.update(action)
            score += reward
            next_action = agent.choose_action(next_state)
            #time.sleep(0.001)
            agent.update(reward, state, action)
            game.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                game.menu.handle_input(event, agent)
            state = next_state
            action = next_action
            if game.game_over:
                print(f"Episode {i}: score={score}")
        if i % 100 == 0:
            agent.set_epsilon(agent.epsilon * 0.95)

    pygame.quit()