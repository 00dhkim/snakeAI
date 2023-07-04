import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, savefig

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

BLOCK_SIZE = 20


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(14, 256, 3) #FIXME: 잠시바꿈
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        head = game.snakes[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = True if game.direction == Direction.LEFT else False
        dir_r = True if game.direction == Direction.RIGHT else False
        dir_u = True if game.direction == Direction.UP else False
        dir_d = True if game.direction == Direction.DOWN else False

        #TODO: 나중에 밖으로 빼서 따로 함수 만들자
        # straight-line obstacle and snake distance
        dist_straight = None
        for obs in game.obstacles + game.snakes[1:]:
            if dir_l:
                # obs 중에서 head보다 왼쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x < head.x and obs.y == head.y:
                    dist_straight = head.x - obs.x
            elif dir_r:
                # obs 중에서 head보다 오른쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x > head.x and obs.y == head.y:
                    dist_straight = obs.x - head.x
            elif dir_u:
                # obs 중에서 head보다 위쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y < head.y:
                    dist_straight = head.y - obs.y
            elif dir_d:
                # obs 중에서 head보다 아래쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y > head.y:
                    dist_straight = obs.y - head.y
        
        if dist_straight is None:
            # head와 벽 끝까지의 거리
            if dir_l:
                dist_straight = head.x
            elif dir_r:
                dist_straight = game.w - head.x
            elif dir_u:
                dist_straight = head.y
            elif dir_d:
                dist_straight = game.h - head.y
            else:
                assert False, "direction error"
        
        # normalize distance
        if dir_l or dir_u:
            dist_straight = dist_straight / game.w / BLOCK_SIZE
        else:
            dist_straight = dist_straight / game.h / BLOCK_SIZE
        
        # right-turn obstacle and snake distance
        dist_right = None
        for obs in game.obstacles + game.snakes[1:]:
            if dir_l:
                # obs 중에서 head보다 위쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y < head.y:
                    dist_right = head.y - obs.y
            elif dir_r:
                # obs 중에서 head보다 아래쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y > head.y:
                    dist_right = obs.y - head.y
            elif dir_u:
                # obs 중에서 head보다 오른쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x > head.x and obs.y == head.y:
                    dist_right = obs.x - head.x
            elif dir_d:
                # obs 중에서 head보다 왼쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x < head.x and obs.y == head.y:
                    dist_right = head.x - obs.x
        if dist_right is None:
            # head와 벽 끝까지의 거리
            if dir_l:
                dist_right = game.h - head.y
            elif dir_r:
                dist_right = head.y
            elif dir_u:
                dist_right = head.x
            elif dir_d:
                dist_right = game.w - head.x
            else:
                assert False, "direction error"
        
        # normalize distance
        if dir_l or dir_d:
            dist_right = dist_right / game.h / BLOCK_SIZE
        else:
            dist_right = dist_right / game.w / BLOCK_SIZE
        
        # left-turn obstacle and snake distance
        dist_left = None
        for obs in game.obstacles + game.snakes[1:]:
            if dir_l:
                # obs 중에서 head보다 아래쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y > head.y:
                    dist_left = obs.y - head.y
            elif dir_r:
                # obs 중에서 head보다 위쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y < head.y:
                    dist_left = head.y - obs.y
            elif dir_u:
                # obs 중에서 head보다 왼쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x < head.x and obs.y == head.y:
                    dist_left = head.x - obs.x
            elif dir_d:
                # obs 중에서 head보다 오른쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x > head.x and obs.y == head.y:
                    dist_left = obs.x - head.x
        if dist_left is None:
            # head와 벽 끝까지의 거리
            if dir_l:
                dist_left = head.y
            elif dir_r:
                dist_left = game.h - head.y
            elif dir_u:
                dist_left = game.w - head.x
            elif dir_d:
                dist_left = head.x
            else:
                assert False, "direction error"
        
        # normalize distance
        if dir_l or dir_u:
            dist_left = dist_left / game.h / BLOCK_SIZE
        else:
            dist_left = dist_left / game.w / BLOCK_SIZE
        
        
        state = [
            # danger straight
            dir_r and game.is_collision(point_r) or \
            dir_l and game.is_collision(point_l) or \
            dir_u and game.is_collision(point_u) or \
            dir_d and game.is_collision(point_d),  # 아래로 가고있는데 아래쪽에 벽이있다? 직진 위험.

            # danger right
            dir_u and game.is_collision(point_r) or \
            dir_d and game.is_collision(point_l) or \
            dir_l and game.is_collision(point_u) or \
            dir_r and game.is_collision(point_d),  # 오른쪽으로 가고있는데 아래쪽에 벽이있다? 우회전 위험.

            # danger left
            dir_d and game.is_collision(point_r) or \
            dir_u and game.is_collision(point_l) or \
            dir_r and game.is_collision(point_u) or \
            dir_l and game.is_collision(point_d),  # 왼쪽으로 가고있는데 아래쪽에 벽이있다? 좌회전 위험.

            # 진행방향에서 head와 장애물간의 거리
            dist_straight,
            dist_right,
            dist_left,

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

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
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # 0 or 1
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # 이렇게 하면 forward 함수가 실행됨
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_recent_mean_scores = []
    total_score = 0
    highest_score = 0
    agent = Agent()
    game = SnakeGameAI()
    
    for episode in range(400):
        while True: # 무한루프이지만, 일정 frame 넘으면 done=True로 바뀜.
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

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

        if score > highest_score:
            highest_score = score
            agent.model.save()

        print(f'Game {agent.n_games} Score {score} Highest Score {highest_score}'
                f' Epsilon {agent.epsilon/200} Memory Length {len(agent.memory)}')

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
