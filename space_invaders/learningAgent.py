import pickle
import random
import time

import matplotlib.pyplot as plt
from IPython import display
import pygame
from space_invaders.game import Game
from space_invaders.GUI import QLearningGUI
from collections import deque


# Create an empty figure and axis


class QLearningAgent:
    def __init__(self, actions, epsilon=0.3, gamma=0.99, alpha=0.01):
        self.q_table = {}  # Q-Tabelle zur Speicherung der Q-Werte
        self.actions = actions  # mögliche Aktionen des Agenten
        self.epsilon = epsilon  # Epsilon für die Epsilon-greedy Strategie
        self.gamma = gamma  # Discount-Faktor
        self.alpha = alpha  # Lernrate
        self.prev_state = None  # vorheriger Zustand des Agenten
        self.prev_action = None  # vorherige Aktion des Agenten
        self.mode = True

    def plot(self, scores):
        display.clear_output(wait=True)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)

        plt.plot(scores)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))

    # plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    # plt.show(block=False)
    # plt.pause(0.001)

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

    # old:
    # def choose_action(self, state, game):
    #     does_player_bullet_exist = game.is_allowed_to_shoot()
    #
    #     if random.random() < self.epsilon:  # Wähle mit einer Wahrscheinlichkeit von Epsilon eine zufällige Aktion
    #         if does_player_bullet_exist:
    #             action = random.choice([1, 2])
    #         else:
    #             action = random.choice(self.actions)
    #     else:  # Wähle sonst die Aktion mit dem höchsten Q-Wert
    #         if does_player_bullet_exist:
    #             available_actions = [1, 2]
    #         else:
    #             available_actions = self.actions
    #
    #         q_values = [self.get_q_value(state, a) for a in available_actions]
    #         max_q_value = max(q_values)
    #         count_max = q_values.count(max_q_value)
    #         if count_max > 1:  # Bei mehreren maximalen Q-Werten, wähle zufällig einen
    #             best_indices = [i for i in range(len(available_actions)) if q_values[i] == max_q_value]
    #             i = random.choice(best_indices)
    #         else:
    #             i = q_values.index(max_q_value)
    #
    #         action = available_actions[i]
    #
    #     return action

    def choose_action(self, state, game):
        does_player_bullet_exist = game.is_allowed_to_shoot()

        if random.random() < self.epsilon:  # Wähle mit einer Wahrscheinlichkeit von Epsilon eine zufällige Aktion
            if does_player_bullet_exist:
                action = random.choice([0, 2, 3])
            else:
                action = 1#random.choice(self.actions)
        else:  # Wähle sonst die Aktion mit dem höchsten Q-Wert
            if does_player_bullet_exist:
                available_actions = [0, 2, 3]
            else:
                available_actions = [1]#self.actions

            q_values = [self.get_q_value(state, a) for a in available_actions]
            max_q_value = max(q_values)
            count_max = q_values.count(max_q_value)
            if count_max > 1:  # Bei mehreren maximalen Q-Werten, wähle zufällig einen
                best_indices = [i for i in range(len(available_actions)) if q_values[i] == max_q_value]
                i = random.choice(best_indices)
            else:
                i = q_values.index(max_q_value)

            action = available_actions[i]

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

    def updateActions(self, game):
        does_player_bullet_exist = game.is_allowed_to_shoot()
        if does_player_bullet_exist:
            self.actions = [1, 2]
        else:
            self.actions = [0, 1, 2, 3, 4]

    def setActions(self, actions):
        self.actions = actions

    def save_q_table(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_path):
        with open(file_path, "rb") as f:
            self.q_table = pickle.load(f)


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Space Invaders")

    game = Game(screen, rows=3, cols=6, game_speed=0.5, enemies_attack=True, enemy_attackspeed=0.01, ai=True)
    agent = QLearningAgent(actions=[0, 1, 2, 3, 4], epsilon=0.042, gamma=.9, alpha=0.0000)

    # agent.load_q_table("collisionBug.pkl")

    # agent.load_q_table("disappearBug.pkl")

    # agent.load_q_table("q_table_new_function_153_108_171.pkl")
    # agent.load_q_table("q_table_new_rewards_372.pkl")

    agent.load_q_table("q_table_new_actions_time.pkl")
    #agent.load_q_table("q_table_new_rewards.pkl")
    # agent.load_q_table("q_table_new_rewards_secondIteration.pkl")

    use_gui = False
    simulation_mode = False  # Hinzufügen der simulation_mode Variable
    # game.run()
    if use_gui and not simulation_mode:
        gui = QLearningGUI(game, agent)
        agent.set_epsilon(gui.epsilon)
        agent.set_gamma(gui.gamma)
        agent.set_alpha(gui.alpha)

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 500)
        mean_scores = []
    else:
        gui = None

    scores = []
    for i in range(1000000):
        game.reset()
        game.game_over = False
        agent.reset()
        state = game.get_state()
        action = agent.choose_action(state, game)
        score = 0

        # agent.epsilon = 0      <------------------------------------------------------------------------------------
        # step = 0
        # print("\n")
        while not game.game_over:
            steps_to_execute = gui.steps_to_execute if use_gui and not simulation_mode else 1
            if steps_to_execute > 0:
                for _ in range(steps_to_execute):
                    # step += 1
                    # if step % 100 == 0:
                    # print(step)
                    if game.game_over:
                        break
                    # agent.updateActions(game)
                    next_state, reward = game.update(action)
                    # print(agent.q_table)
                    score += reward
                    next_action = agent.choose_action(next_state, game)
                    agent.update(reward, state, action)
                    # time.sleep(.01)
                    # Zeichnen und GUI-Aktualisierung nur, wenn simulation_mode deaktiviert ist

                    if not simulation_mode or (gui is not None and gui.game_draw_enabled):
                        game.draw(agent=agent)
                        time.sleep(0.01)
                        if use_gui:
                            # gui.update()
                            gui.root.update()

                    state = next_state
                    action = next_action
                    if game.game_over:
                        # scores.append(score)
                        if not simulation_mode and gui is not None and gui.game_draw_enabled:
                            print(f"Episode {i}: score={score}")
                if use_gui and not simulation_mode and gui.game_draw_enabled:
                    gui.steps_to_execute = 0
                    action = next_action
            else:
                if use_gui and not simulation_mode:
                    # gui.update()
                    gui.root.update()

        # agent.set_epsilon(agent.epsilon * 0.99999)                   <--------------------------------------RESET
        # agent.set_alpha(agent.alpha * 0.99999)
        # print(agent.q_table)
        if i % 3000 == 0 and i != 0:
            print("Updating q_table \nIn episode: " + str(i))
            print("Current e: " + str(agent.epsilon) + "\n\n\n")

            agent.save_q_table("q_table_new_rewards_secondIteration.pkl")
            agent.plot(scores)
            plt.savefig('training_plot.png')

            if use_gui:
                plt.savefig('training_plot.png')
                plt.show(block=False)
                plt.pause(.001)

    pygame.quit()


def main2():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Space Invaders")

    game = Game(screen, rows=3, cols=6, game_speed=0.5, enemies_attack=True, enemy_attackspeed=0.01, ai=True)
    agent = QLearningAgent(actions=[0, 1, 2, 3, 4], epsilon=1, gamma=.9, alpha=0.0001)

    #agent.load_q_table("q_table_new_actions_time.pkl")

    simulation_mode = False

    scores = deque(maxlen=3000)
    averages = []
    for i in range(1000000):
        game.reset()
        game.game_over = False
        agent.reset()
        state = game.get_state()
        action = agent.choose_action(state, game)
        score = 0

        while not game.game_over:
                for _ in range(30000):
                    if game.game_over:
                        break
                    next_state, reward = game.update(action)
                    score += reward
                    next_action = agent.choose_action(next_state, game)
                    agent.update(reward, state, action)

                    if not simulation_mode:
                        game.draw(agent=agent)
                        time.sleep(0.01)

                    state = next_state
                    action = next_action
                    if game.game_over:
                        if not simulation_mode:
                            print(f"Der Agent hat einen Score von: {score} erreicht!")
                        scores.append(score)
                        if not simulation_mode:
                            print(f"Episode {i}: score={score}")

        agent.set_epsilon(agent.epsilon * 0.99999)

        if i % 3000 == 0 and i != 0:
            print("Updating q_table \nIn episode: " + str(i))
            print("Current e: " + str(agent.epsilon) + "\n\n\n")

            average = sum(scores) / len(scores)
            averages.append(average)
            agent.save_q_table("q_table_newTEST.pkl")
            agent.plot(averages)
            plt.savefig('training_plot.png')

    pygame.quit()


if __name__ == "__main__":
    main2()
