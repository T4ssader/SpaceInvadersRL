import random
import matplotlib.pyplot as plt
from IPython import display

from space_invaders.GUI import QLearningGUI

# Create an empty figure and axis
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Episode')
ax.set_ylabel('Score')
ax.set_xlim(0, 1000)
ax.set_ylim(0, 500)
mean_scores = []

class QLearningAgent:
    def __init__(self, actions, epsilon=0.3, gamma=0.99, alpha=0.01):
        self.q_table = {}  # Q-Tabelle zur Speicherung der Q-Werte
        self.actions = actions  # mögliche Aktionen des Agenten
        self.epsilon = epsilon  # Epsilon für die Epsilon-greedy Strategie
        self.gamma = gamma  # Discount-Faktor
        self.alpha = alpha  # Lernrate
        self.prev_state = None  # vorheriger Zustand des Agenten
        self.prev_action = None  # vorherige Aktion des Agenten

    def plot(self, scores):
        display.clear_output(wait=True)
        #display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        sum_score = 0
        for score in scores:
            sum_score += score
        mean_scores.append(sum_score / len(scores))

        plt.plot(mean_scores)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

        plt.show(block=False)
        plt.pause(.001)

    def get_q_value(self, state, action):
        state = tuple(state)  # Konvertiere Zustand in ein Tuple
        if state not in self.q_table:  # Wenn der Zustand noch nicht in der Q-Tabelle ist, füge ihn hinzu
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if action not in self.q_table[
            state]:  # Wenn die Aktion noch nicht in der Q-Tabelle für diesen Zustand ist, füge sie hinzu
            self.q_table[state][action] = 0.0
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

    # Ich brauch eine Funktion namens updateActions, die die möglichen Aktionen des Agenten ändert
    # Es soll geschaut werden ob auf dem Spielfeld ein bullet vom spieler ist, wenn ja, dann
    # sollen die actionen 0, 3, 4 nicht mehr möglich sein und wenn nicht, dann sollen sie wieder möglich sein
    def updateActions(self, game):
        does_player_bullet_exist = False
        for bullet in game.bullets:
            if bullet.player_bullet:
                does_player_bullet_exist = True
                self.actions = [1, 2]
                break
        if not does_player_bullet_exist:
            self.actions = [0, 1, 2, 3, 4]

    def setActions(self, actions):
        self.actions = actions


import pygame
from space_invaders.game import Game

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Space Invaders")

    scores = []

    game = Game(screen, rows=3, cols=6, game_speed=0.5, enemies_attack=True, enemy_attackspeed=0.01, ai=True)
    agent = QLearningAgent(actions=[0, 1, 2, 3, 4], epsilon=0.15, gamma=1, alpha=0.1)
    gui = QLearningGUI(game, agent)

    # Aktualisieren Sie die Agentenparameter basierend auf den GUI-Werten
    agent.set_epsilon(gui.epsilon)
    agent.set_gamma(gui.gamma)
    agent.set_alpha(gui.alpha)

    for i in range(10000):
        game.reset()
        game.game_over = False
        agent.reset()
        state = game.get_state()
        action = agent.choose_action(state)

        score = 0
        # print the three variables
        #print("Episode: ", i, "Epsilon: ", agent.epsilon, "Alpha: ", agent.alpha, "Gamma: ", agent.gamma)
        while not game.game_over:
            if gui.steps_to_execute > 0:
                for _ in range(gui.steps_to_execute):
                    if game.game_over:
                        break
                    agent.updateActions(game)
                    next_state, reward = game.update(action)
                    score += reward
                    next_action = agent.choose_action(next_state)
                    # actions = 0 = shoot, 1 = left, 2 = right, 3 = leftShoot, 4 = rightShoot
                    # print action and the name of the action
                    #print(action)
                    agent.update(reward, state, action)
                    game.draw(agent=agent)
                    gui.root.update()
                    state = next_state
                    action = next_action
                    #print("actions: ", agent.actions)
                gui.steps_to_execute = 0
                if game.game_over:
                    scores.append(score)
                    print(f"Episode {i}: score={score}")
            else:
                gui.root.update()

        agent.plot(scores)

        if i % 100 == 0:
            gui.set_epsilon(agent.epsilon * 0.95)

    plt.show()
    pygame.quit()