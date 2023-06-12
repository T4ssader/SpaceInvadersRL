import gc

import cv2
import numpy as np
import random

import pygame
import pylab
from keras.backend import clear_session
from matplotlib import pyplot as plt

from collections import deque
from space_invaders.game2 import Game
from PER import *

# All the Keras imports
from keras.backend import clear_session
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K


def OurModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input

    # This is the convolutional layers of the network
    X = Conv2D(64, 5, strides=(3, 3), padding="valid", input_shape=input_shape, activation="relu",
               data_format="channels_first")(X)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Flatten()(X)

    # This is the fully connected layer of the network
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # The output layer of the network
    if dueling:
        # Dueling Network
        # It separates the value and advantage streams of the network.
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    # Compile the model with loss function and optimizer
    model = Model(inputs=X_input, outputs=X, name='CartPole_model')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()  # Prints the summary of the model
    return model  # Returns the model object


class CNNSpaceInvaders:
    def __init__(self):
        self.env_name = "SpaceInvaders"
        self.TAU = 0.1  # target network soft update hyperparameter
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0001
        self.gamma = 0.95

        self.scores, self.episodes, self.average = [], [], []

        self.EPISODES = 1000
        self.ROWS = 160
        self.COLS = 210
        self.REM_STEP = 4
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Space Invaders")
        self.env = Game(self.screen, rows=3, cols=6, game_speed=1, enemies_attack=True, enemy_attackspeed=0.01, ai=True)

        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.action_size = 6

        self.USE_PER = True  # use priority experienced replay
        self.dueling = True  # use dealing netowrk
        self.ddqn = True  # use doudle deep q network
        self.Soft_Update = False  # use soft parameter update
        self.epsilon_greedy = False

        # Setting up memory and other parameters
        memory_size = 10000
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=2000)

        self.batch_size = 64

        self.model = OurModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)
        self.target_model = OurModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)

    def GetImage(self, e):
        img = self.env.render()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        # img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis=0)
        self.image_memory[0, :, :] = img_rgb_resized

        # self.imshow(self.image_memory,0)

        if e % 1000 and e is not 0:
            # save the image
            plt.imsave("image1000.png", self.image_memory[0, :, :], cmap="gray")


        return np.expand_dims(self.image_memory, axis=0)

    def reset(self):
        self.env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage()
        return state

    def step(self, action,e):
        reward, done = self.env.step(action)
        next_state = self.GetImage(e)
        return next_state, reward, done

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            if self.env.is_allowed_to_shoot():
                return random.choice([0, 1, 2]), explore_probability
            else:
                return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            act_values = self.model.predict(state, verbose=0)[0]
            if self.env.is_allowed_to_shoot():
                return np.argmax(act_values[:3]), explore_probability
            else:
                return np.argmax(act_values), explore_probability

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = load_model(name)

    # pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episode', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn: dqn = 'DDQN_'
        if self.Soft_Update: softupdate = '_soft'
        if self.dueling: dueling = '_Dueling'
        if self.epsilon_greedy: greedy = '_Greedy'
        if self.USE_PER: PER = '_PER'
        try:
            pylab.savefig(dqn + self.env_name + softupdate + dueling + greedy + PER + "_CNN.png")
        except OSError:
            pass

        return str(self.average[-1])[:5]

    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def replay(self):
        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size,) + self.state_size)
        next_state = np.zeros((self.batch_size,) + self.state_size)
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(len(minibatch)):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.model.predict(state, verbose=0)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state, verbose=0)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state, verbose=0)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                if self.ddqn:  # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

            if self.USE_PER:
                absolute_errors = np.abs(target_old[i] - target[i])
                # Update priority
                self.MEMORY.batch_update(tree_idx, absolute_errors)

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def run(self):
        decay_step = 0
        for e in range(self.EPISODES):
            state = self.reset()  # returns
            done = False
            i = 0
            while not done:
                decay_step += 1
                action, explore_probability = self.act(state)
                next_state, reward, done = self.step(action, i)
                if not done:
                    reward = reward
                else:
                    reward = -500
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every REM_STEP update target model
                    if e % self.REM_STEP == 0:
                        self.update_target_model()
                        gc.collect()
                        clear_session()

                    # every episode, plot the result
                    average = self.PlotModel(i, e)

                    print("episode: {}/{}, steps: {}, e: {:.2}".format(e, self.EPISODES, i,
                                                                       explore_probability))
                    if e % 50 == 0:
                        print("Saving trained model to: ", "1CNN.h5")
                        self.save("1CNN.h5")
                if i % 4 == 0:
                    self.replay()

    def test(self):
        self.reset()

    def play_model(self):
        # in this method we want to let the agent play with an epsilon of 0 so we see how good he is
        self.epsilon = 0




if __name__ == '__main__':
    agent = CNNSpaceInvaders()
    agent.run()
    # agent.test()
    # agent.play_model()
