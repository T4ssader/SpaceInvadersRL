import gc

import gymnasium as gym
import numpy as np
import random

import tensorflow as tf
from collections import deque
import gc

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.65  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = self.model.predict_on_batch(states)
        Q_future = np.amax(self.target_model.predict_on_batch(next_states), axis=1)
        targets[range(batch_size), actions] = rewards + self.gamma * (1 - dones) * Q_future

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    play = True
    if play:
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('CartPole-v1', render_mode="rgb_array")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 128
    episodes = 500
    steps = 0
    steps_average = 1
    if not play:
        for e in range(episodes):
            if e != 0:
                print("\n\nEpisode: {}/{}, e: {:.2}, steps_avg: {} ".format(e, episodes, agent.epsilon, steps_average/e))
            steps = 0
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            for step in range(1500):
                steps += 1
                steps_average +=1
                if step == 499:
                    print("STEPS MAX")
                env.render()
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward = reward if not terminated or truncated else -1
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, terminated or truncated)
                state = next_state
                if terminated or truncated:
                    break
                if len(agent.memory) > batch_size and (step % 5 == 0):
                    agent.replay(batch_size)
                    agent.update_target_model()
                    gc.collect()
                    clear_session()
            if e % 25 == 0:
                agent.update_target_model()
                agent.model.save_weights('v1Test_RGB_{}.h5'.format(e))
    else:
        agent.epsilon = 0
        weights_filename = 'CartpoleGYM_RGB_400e.h5'
        agent.model.load_weights(weights_filename)
        for e in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            while True:
                env.render()
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = np.reshape(next_state, [1, state_size])
                if terminated or truncated:
                    break


if __name__ == "__main__":
    # bestrafung wieder auf -10 machen und eventuall epsilon schneller fallen lassen?
    # 'v1Test_400.h5' gut, nicht perfekt!
    # 'CartpoleGYM_RGB_400e.h5' ist perfekt!
    main()
