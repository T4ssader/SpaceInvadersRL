import gc
import random
from collections import deque

import numpy as np
import pygame
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from space_invaders.game import Game


class DQNv2:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.65  # discount factor
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.999999
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        #model = Sequential()
        #model.add(Dense(128, input_dim=self.state_space, activation='relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(self.action_space, activation='linear'))
        #model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        #return model

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, env):
        # Check if the player bullet exists
        does_player_bullet_exist = env.is_allowed_to_shoot()
        act_values = self.model.predict(state, verbose=0)[0]
        if np.random.rand() <= self.epsilon:
            if does_player_bullet_exist:
                return random.choice([0, 1, 2])
            else:
                return random.randrange(self.action_size)

        if does_player_bullet_exist:
            return np.argmax(act_values[:3])
        else:
            return np.argmax(act_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, self.batch_size)

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
    draw_15 = True
    draw_always = False
    pygame.init()

    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Space Invaders")
    env = Game(screen, rows=3, cols=6, game_speed=1, enemies_attack=True, enemy_attackspeed=0.01, ai=True)

    state_size = 21
    action_size = 6
    agent = DQNv2(state_size, action_size)
    batch_size = 128
    episodes = 5000
    steps = 0
    steps_average = 1

    if not play:
        for e in range(episodes):
            if e != 0:
                print("\n\nEpisode: {}/{}, e: {:.2}, steps_avg: {:.4} ".format(e, episodes, agent.epsilon,
                                                                               steps_average / e))
            steps = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for step in range(5000000):
                steps += 1
                steps_average += 1
                if step == 4999999:
                    print("STEPS MAX")
                if (draw_15 and e % 15 == 0) or draw_always:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                    env.render()
                action = agent.act(state, env)

                next_state, reward, terminated = env.step(action)
                #print("Reward: ", reward)
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, terminated)
                state = next_state
                if terminated:
                    break
                if len(agent.memory) > batch_size and (step % 5 == 0):
                    agent.replay(batch_size)
                    agent.update_target_model()
                    gc.collect()
                    clear_session()
            if e % 25 == 0:
                agent.update_target_model()
                agent.model.save_weights('SpaceInvTestv7_{}.h5'.format(e))
    else:
        agent.epsilon = 0
        weights_filename = 'SpaceInvTestv7_275.h5'
        agent.model.load_weights(weights_filename)
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                env.render()
                action = agent.act(state,env)
                #print(action)
                next_state, reward, terminated = env.step(action)
                state = np.reshape(next_state, [1, state_size])
                if terminated:
                    break


if __name__ == '__main__':
    main() #SpaceInvTestv4_125.h5 ist okay aber noch fehler








