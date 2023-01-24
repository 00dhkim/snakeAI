#!/usr/bin/env python
# coding: utf-8

# In[1]:


from snake_gym import SnakeGym


# In[2]:


import sys
import random
import numpy as np
from collections import deque

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

import logging

# In[3]:

EPISODES = 10000
EPISODE_LENGTH = 200

LOAD_MODEL = False
LOAD_MODEL_PATH = './Python/snake_dqn_1.h5'

RENDER = False
RENDER_DELAY = 0.1

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = RENDER
        self.load_model = LOAD_MODEL

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.10
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 5000
        self.memory = deque(maxlen=5000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights(LOAD_MODEL_PATH)
    
    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        
    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            q_value = self.model.predict(state, verbose=0)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))
        
        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

# In[5]:

logging.basicConfig(filename='log.log', level=logging.DEBUG, filemode='w',
                    format="%(asctime)s %(message)s")
logging.info('logging start\n')

env = SnakeGym()
state_size = env.state_size
action_size = env.action_size

agent = DQNAgent(state_size, action_size)
scores, episodes = [], []

save_flag_1, save_flag_2 = True, True

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset() # (3, 10, 10)
    state = np.reshape(state, [1, state_size]) # (1, 300)
    
    for _ in range(EPISODE_LENGTH):
        if RENDER:
            print(f'episode: {e}')
            env.render(delay=RENDER_DELAY)
        
        # 현재 상태로 행동을 선택
        action = agent.get_action(state)
        
        # 선택한 행동으로 환경에서 한 타임스텝 진행
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size]) # (3, 10, 10) -> (1, 300)
        
        agent.append_sample(state, action, reward, next_state, done)
        
        # 매 타임스텝마다 학습
        if len(agent.memory) > agent.train_start:
            agent.train_model()
        
        score += reward
        state = next_state
        
        if done:
            break
    
    agent.update_target_model()
    scores.append(score)
    episodes.append(e)
    # pylab.plot(episodes, scores, 'b')
    # pylab.savefig('./Python/snake_dqn.png')
    
    print(f"episode:{e} score: {score:+} snklen: {info['snake_length']} "
            f"memlen: {len(agent.memory)} epsilon: {agent.epsilon}")
    # logging.info(f"episode:{e} score: {score:+} snklen: {info['snake_length']} "
    #              f"memlen: {len(agent.memory)} epsilon: {agent.epsilon}")
    
    # # 최근 10 에피소드의 점수 평균이 -30 이상일 때 저장
    # if save_flag_1 and np.mean(scores[-min(10, len(scores)):]) > -30:
    #     agent.model.save_weights("./Python/snake_dqn_1.h5")
    #     print('model 1 saved at', e)
    #     save_flag_1 = False
    
    # # 최근 10 에피소드의 점수 평균이 10 이상일 때 저장
    # if save_flag_2 and np.mean(scores[-min(10, len(scores)):]) > 10:
    #     agent.model.save_weights("./Python/snake_dqn_2.h5")
    #     print('model 2 saved at', e)
    #     save_flag_2 = False

    # # 최근 10 에피소드의 점수 평균이 30 이상일 때 종료
    # if np.mean(scores[-min(10, len(scores)):]) > 30:
    #     agent.model.save_weights("./Python/snake_dqn_3.h5")
    #     print('model 3 saved at', e)
    #     sys.exit()
    
    # 뱀의 꼬리가 3 이상일 때 저장
    if save_flag_1 and info['snake_length'] > 3:
        agent.model.save_weights("./Python/snake_dqn_1.h5")
        print('model 1 saved at', e)
        save_flag_1 = False
    
    # 뱀의 꼬리가 5 이상일 때 저장
    if save_flag_2 and info['snake_length'] > 5:
        agent.model.save_weights("./Python/snake_dqn_2.h5")
        print('model 2 saved at', e)
        save_flag_2 = False
    
    # 뱀의 꼬리가 10 이상일 때 종료
    if info['snake_length'] > 10:
        agent.model.save_weights("./Python/snake_dqn_3.h5")
        print('model 3 saved at', e)
        sys.exit()
    
            
# %%
