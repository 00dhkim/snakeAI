import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer, Linear_QNet2
from helper import plot, savefig

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
MINIMUM_EPSILON = 0.00

BLOCK_SIZE = 20

''' 방향에 대한 개념

- 왼쪽 위가 (0, 0)
- 가로 길이는 w, 세로 길이는 h
- x는 오른쪽, y는 아래쪽으로 갈수록 증가

+---------- w [j]
|0         
|          
|        →x
|    ↓     
|    y     
h [i]

e.g.,
Point(x, y)
Point(j, i)
Point(w, h)
'''

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        # self.model = Linear_QNet(11+50, 256, 3) #TODO: input_size: state + window
        self.model = Linear_QNet2(11, 256, 3) # input_size: state + window
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        # 즉, 오래된 거부터 버림
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # 주석처리된 부분은 위에랑 기능이 도일함.
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(np.array(state), action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.5 - self.n_games/200, MINIMUM_EPSILON)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)  # 0 ~ 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # 이렇게 하면 forward 함수가 실행됨
            move = torch.argmax(prediction)
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_recent_mean_scores = []
    total_score = 0
    highest_score = 0
    agent = Agent()
    game = SnakeGame()
    
    for episode in range(1000):
        while True: # 무한루프이지만, 일정 frame 넘으면 done=True로 바뀜.
            # get old state
            state_old = game.get_state()

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = game.get_state()

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                break
        
        # train long memory, plot result
        game.reset()
        agent.n_games += 1
        agent.train_long_memory()
        agent.trainer.lr_scheduler.step() ## LR 스케줄링

        if score > highest_score:
            highest_score = score
            agent.model.save()

        lr = agent.trainer.lr_scheduler.get_last_lr()
        print(f'Game {agent.n_games:3} Score {score:2} Highest Score {highest_score:2}'
              f' Epsilon {agent.epsilon:4} Memory Length {len(agent.memory):6}'
              f' LR {lr}')

        plot_scores.append(score)
        total_score += score
        # mean_score = total_score / agent.n_games
        recent_mean_score = np.mean(plot_scores[-50:])
        plot_recent_mean_scores.append(recent_mean_score)
        plot(plot_scores, plot_recent_mean_scores)

    # 학습종료
    savefig('Python\\freeCodeCamp\\plot.png')


if __name__ == '__main__':
    train()
