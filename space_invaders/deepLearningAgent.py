import os.path
import time

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, Flatten, Dense
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import pickle

from space_invaders.game import Game
import pygame

pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Space Invaders")
env = Game(screen, rows=3, cols=6, game_speed=0.001, enemies_attack=True, enemy_attackspeed=0.01, ai=True)


class DQN:
    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.9
        self.gamma = .97
        self.batch_size = 100000
        self.epsilon_min = .005
        self.epsilon_decay = .999
        self.learning_rate = 0.01
        self.memory = deque(maxlen=100000)

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        act_values = self.model.predict(state, verbose=0)[0]

        best_actions = np.argwhere(act_values == np.amax(act_values))

        best_actions = best_actions.flatten().tolist()
        return random.choice(best_actions)


    def replay(self):
        if len(self.memory) < self.batch_size:
            #print("\n\n\nmemory overrun\nFinishing...")
            return

        minibatch = random.sample(self.memory, self.batch_size)
        # print("minibatch: " + str(minibatch))
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # env.draw()

    def play_with_model(self, model_path):
        self.model = load_model(model_path)
        self.learning_rate = 0
        self.epsilon = 0
        self.epsilon_decay = 0

        state_space = 21
        done = False



        for _ in range(10):
            state = env.reset()
            state = np.reshape(state, (1, state_space))
            done = False
            while not done:
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         done = True

                next_action = self.act(state)
                next_state, reward, done = env.step(next_action)
                next_state = np.reshape(next_state, (1, state_space))
                state = next_state
                #time.sleep(0.0005)
                env.draw()


def train_dqn(episode, model_name):
    if os.path.isdir(model_name):
        a = input(f"You are about to overrite \"{model_name}\", are you sure about that?\n")
        if a.lower() not in ["yes","y","ja","j"]:
            print("Canceled")
            exit()
        else:
            print("\nSuccessfull, continuing")

    loss = []
    nb_score = []

    action_space = 5
    state_space = 21
    max_steps = 900100
    record = 0
    score = 0

    agent = DQN(action_space, state_space)
    agent.model.save("EMPTY_MODEL")
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0
        itera = 0
        print()
        while itera < max_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            score += reward
            # TODO change to not save every record
            if score > record:
                record = score
                print("new record = ", score)
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
            itera += 1
        if max_steps < itera:
            print("Stopped because too many steps")
        if e != 0 and (e % 100) == 0:
            agent.model.save(model_name)
            print("saved episode = ", e)

        loss.append(score)

        # Quit game
        # if e > episode :
        # sys.exit()

    agent.model.save(model_name)

    return loss, agent


if __name__ == '__main__':
    train = True
    model_name = "model_1000e"

    if train:
        ep = 100
        print(tf.config.list_physical_devices('GPU'))
        loss, model = train_dqn(ep, model_name)


        with open("model_file.pkl", "wb") as binary_file:
            pickle.dump(model, binary_file, pickle.HIGHEST_PROTOCOL)
        plt.plot([i for i in range(len(loss))], loss, label="loss")
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.legend()
        plt.show()

    else:
        agent = DQN(5, 21)
        agent.play_with_model(model_name)