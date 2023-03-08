"""
snake_gym.py

snake 게임의 플레이 기능 담은 소스코드

"""


import os
import random
import numpy as np
import time
from enum import Enum
from typing import List, Tuple

MAP_SIZE = 10

REWARD_EAT = 10 # 길어지도록 유도
REWARD_ALIVE = 0 # 
REWARD_DEAD = -10 # 벽이나 몸통에 닿지 않도록 유도

UP = [1, 0, 0, 0]
RIGHT = [0, 1, 0, 0]
DOWN = [0, 0, 1, 0]
LEFT = [0, 0, 0, 1]

EMPTY = 0
SNAKE = 1
FOOD = 2
WALL = 3

class Snake:
    # single agent now
    '''
    locations: [(int, int)]
        snake 위치를 나타내는 리스트, (i, j)
    
    health: int
        100으로 초기화, 매 step -1, 0이 되면 사망
        TODO: health 개념 비활성화
    
    map_size: int
        전체 지도는 가로 세로가 각각 map_size인 정사각형 형태
    '''

    def __init__(self, map_size, starting_pos=(0, 0)):
        self.map_size = map_size
        self.locations = []  # locations[0]: head
        self.locations.append(starting_pos)
        self.health = 100

    def insert_head(self, pos):
        self.locations.insert(0, pos)
    
    def pop_tail(self):
        return self.locations.pop()
    
    def head_pos(self):
        return self.locations[0]
    
    # TODO: health 개념 비활성화
    # def health_decrease(self):
    #     self.health -= 1
    #     if self.health <= 0: # dead
    #         return 'dead'
    #     else:
    #         return self.health
    
    # TODO: health 개념 비활성화
    # def heal(self):
    #     self.health = 100
    
    # def __getitem__(self, index):
    #     return self.locations[index]

class SnakeGym:
    
    def __init__(self, map_size=MAP_SIZE):
        self.state_size = 12
        self.action_size = 4
        self.map_size = map_size
        self.step_len = 0
        self.eat_cnt = 0

        self.snake = Snake(map_size=map_size) # Snake class
        self.foods = []  # [(int, int)]
        self.map = None  # [(int, int)]
        self.last_action: list[int] = None

    def reset(self):
        self.step_len = 0
        self.eat_cnt = 0
        starting_pos = (random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2))
        self.last_action = UP
        
        del self.snake
        self.snake = Snake(map_size=self.map_size,
                           starting_pos=starting_pos)
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.int32)

        # generate the wall
        for i in range(self.map_size):
            self.map[0][i] = WALL
            self.map[i][0] = WALL
            self.map[self.map_size - 1][i] = WALL
            self.map[i][self.map_size - 1] = WALL

        # starting posision of snake
        self.map[starting_pos[0]][starting_pos[1]] = SNAKE

        self._set_food()
        # self._set_obstacle()
        
        return self._get_state()

    def _set_food(self):
        """
        랜덤 위치에 먹이를 둠, 매 턴마다 수행
        """
        while True:
            i, j = random.randint(
                1, self.map_size - 2), random.randint(1, self.map_size - 2)
            if self.map[i][j] == 0:
                break
        self.map[i][j] = 2
        self.foods.append((i, j))

    def _set_obstacle(self):
        """
        랜덤 위치에 장애물을 둠, 처음에만 수행
        """
        while True:
            i, j = random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)
            if self.map[i][j] == 0:
                break
        self.map[i][j] = 3

    def step(self, action: list[int]):
        """
        움직이게 될 자리에 뭐가 있는지 체크
        - 벽이 있다면, 사망
        - 먹이가 있다면, 성장
        그 후, 이동
        """
        self.step_len += 1
        reward = REWARD_ALIVE
        done = False
        grow = False

        if np.array_equal(action, UP):
            direc = [-1, 0]
        elif np.array_equal(action, RIGHT):
            direc = [0, 1]
        elif np.array_equal(action, DOWN):
            direc = [1, 0]
        elif np.array_equal(action, LEFT):
            direc = [0, -1]
        else:
            raise ValueError('input must be like [1, 0, 0, 0] style')
        self.last_action = action

        i, j = self.snake.head_pos()[0] + direc[0], self.snake.head_pos()[1] + direc[1]
        
        # TODO: health 개념 비활성화
        # if self.snake.health_decrease() == 'dead': # 체력 없다면
        #     done = True
        #     reward = REWARD_DEAD
        if self.map[i][j] == FOOD: # FOOD 먹었다면
            grow = True
            reward += REWARD_EAT
            self.foods.remove((i, j))
            self._set_food()
            # self.snake.heal()
            self.eat_cnt += 1
        elif self.map[i][j] == SNAKE or self.map[i][j] == WALL: # SNAKE 또는 WALL 부딪혔다면
            reward = REWARD_DEAD
            done = True

        # update snake coordinates
        self.snake.insert_head((i, j))
        self.map[i][j] = SNAKE
        if not grow:
            i, j = self.snake.pop_tail()
            self.map[i][j] = 0

        return self._get_state(), reward, done, {'step_length': self.step_len,
                                                 'snake_length': self.snake.locations.__len__(),
                                                 'snake_health': self.snake.health,
                                                 'eat_cnt': self.eat_cnt,
                                                 }
    
    def _get_state(self):
        '''
        [danger up, danger right,
        danger down, danger left,

        direction up, direction right,
        direction down, direction left,

        # 음식이 여러개 있는 경우, 가장 가까운 음식을 기준으로
        food up, food right,
        food down, food left
        ]
        '''
        state = np.zeros(self.state_size, dtype=np.int32)
        head = self.snake.head_pos()
        
        head_u = (head[0] - 1, head[1])
        head_r = (head[0], head[1] + 1)
        head_d = (head[0] + 1, head[1])
        head_l = (head[0], head[1] - 1)
        
        # danger up
        state[0] = self._is_collision(head_u)
        
        # danger right
        state[1] = self._is_collision(head_r)
        
        # danger down
        state[2] = self._is_collision(head_d)
        
        # danger left
        state[3] = self._is_collision(head_l)
        
        # direction
        state[4:8] = [self.last_action == UP,
                      self.last_action == RIGHT,
                      self.last_action == DOWN,
                      self.last_action == LEFT]
        
        # 머리에서 가장 가까운 food 찾기
        closest_food = None
        for food in self.foods:
            if closest_food is None:
                closest_food = food
            else:
                if abs(head[0] - food[0]) + abs(head[1] - food[1]) < abs(head[0] - closest_food[0]) + abs(head[1] - closest_food[1]):
                    closest_food = food
        
        # food
        state[8] = closest_food[0] < head[0] # food up
        state[9] = closest_food[1] > head[1] # food right
        state[10] = closest_food[0] > head[0] # food down
        state[11] = closest_food[1] < head[1] # food left
        
        return state
    
    def _is_collision(self, pt: tuple=None):
        """지금 pt 위치 기준 부딪혔는지 여부
        """
        if pt is None:
            pt = self.snake.head_pos()
        i, j = pt
        if i < 0 or i >= self.map_size or j < 0 or j >= self.map_size:
            return True # 맵 밖으로 나가면 부딪힌 것으로 간주
        elif self.map[i][j] in [SNAKE, WALL]:
            return True
        else:
            return False
    
    def render(self, delay: float = 0.0):
        _ = os.system('cls' if os.name == 'nt' else 'clear')

        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[i][j] == EMPTY:
                    print(' ', end=' ')
                elif self.map[i][j] == SNAKE:
                    print('O', end=' ')
                elif self.map[i][j] == FOOD:
                    print('*', end=' ')
                if self.map[i][j] == WALL:
                    print('#', end=' ')
            print()
        
        if delay:
            time.sleep(delay)

    def _char2idx(self, char):
        if char == 'w':
            return UP
        elif char == 'd':
            return RIGHT
        elif char == 's':
            return DOWN
        elif char == 'a':
            return LEFT
        else:
            print('input must be 0, 1, 2, 3')
            return 'invalidInput'


if __name__ == '__main__':
    game = SnakeGym()
    game.reset()

    while True:
        while True:
            print('input>', end='', flush=True)
            action = game._char2idx(input())
            if action != 'invalidInput':
                break

        state, reward, done, info = game.step(action)
        if done:
            print("Game Over")
            break
            # game.reset()

        game.render()
