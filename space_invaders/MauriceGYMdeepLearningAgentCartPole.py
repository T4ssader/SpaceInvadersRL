import gymnasium as gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                Q_future  = self.model.predict(next_state, verbose=0)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    play = False
    if play:
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('CartPole-v1', render_mode="rgb_array")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 16
    episodes = 10

    if not play:
        for e in range(episodes):
            print("\n\nEpisode: {}/{}, e: {:.2}".format(e, episodes, agent.epsilon))
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            for step in range(500):
                #env.render()
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward = reward if not terminated or truncated else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, terminated or truncated)
                state = next_state
                if terminated or truncated:
                    break
                if len(agent.memory) > batch_size and (step % 5 == 0):
                    agent.replay(batch_size)
            if e % 25 == 0:
                agent.model.save_weights('model_weights_episode_{}.h5'.format(e))
    else:
        weights_filename = 'weights_200.h5'
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
    main()
