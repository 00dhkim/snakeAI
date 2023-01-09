"""
snake_gym.py

snake 게임의 플레이 기능 담은 소스코드

"""


import os
import random
from tkinter import LEFT
from tracemalloc import start
import numpy as np
import time

MAP_SIZE = 10


class Snake:
    # single agent now
    '''
    locations: [(int, int)]
        snake 위치를 나타내는 리스트, (i, j)
    
    health: int
        100으로 초기화, 매 step -1, 0이 되면 사망
    
    map_size: int
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
    
    def health_decrease(self):
        self.health -= 1
        if self.health <= 0: # dead
            return 'dead'
        else:
            return self.health
    
    def heal(self):
        self.health = 100
    
    # def __getitem__(self, index):
    #     return self.locations[index]

class SnakeGym:

    # directions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    # map labels
    EMPTY = 0
    SNAKE = 1
    FOOD = 2
    WALL = 3

    def __init__(self, map_size=MAP_SIZE):
        '''TODO:
        rewards
        - eat   : 0
        - alive : +1
        - dead  : -100
        '''
        self.state_size = map_size * map_size * 3 # num. of channels
        self.action_size = 4
        self.map_size = map_size
        self.step_len = 0
        self.eat_cnt = 0

        self.snake = Snake(map_size=map_size) # Snake class
        self.foods = []  # [(int, int)]
        self.map = None  # [(int, int)]

    def reset(self):
        self.step_len = 0
        self.eat_cnt = 0
        starting_pos = (random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2))
        
        del self.snake
        self.snake = Snake(map_size=self.map_size,
                           starting_pos=starting_pos)
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.int32)

        # generate the wall
        for i in range(self.map_size):
            self.map[0][i] = self.WALL
            self.map[i][0] = self.WALL
            self.map[self.map_size - 1][i] = self.WALL
            self.map[i][self.map_size - 1] = self.WALL

        # starting posision of snake
        self.map[starting_pos[0]][starting_pos[1]] = self.SNAKE

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

    def step(self, action):
        """
        움직이게 될 자리에 뭐가 있는지 체크
        - 벽이 있다면, 사망
        - 먹이가 있다면, 성장
        그 후, 이동
        """
        self.step_len += 1
        reward = 1
        done = False
        grow = False

        if action == 0:   # up
            direc = [-1, 0]
        elif action == 1:  # right
            direc = [0, 1]
        elif action == 2:  # down
            direc = [1, 0]
        elif action == 3:  # left
            direc = [0, -1]
        else:
            raise ValueError('input must be 0, 1, 2, 3')

        i, j = self.snake.head_pos()[0] + direc[0], self.snake.head_pos()[1] + direc[1]
        
        if self.snake.health_decrease() == 'dead': # 체력 없다면
            done = True
            reward -= 100
        elif self.map[i][j] == self.FOOD: # FOOD 먹었다면
            grow = True
            # reward += 1
            reward += 0 # 일단 살아있는 모델부터 만들어보자. 지금은 자꾸 죽어버림.
            self.foods.remove((i, j))
            self._set_food()
            self.snake.heal()
            self.eat_cnt += 1
        elif self.map[i][j] == self.SNAKE or self.map[i][j] == self.WALL: # SNAKE 또는 WALL 부딪혔다면
            reward -= 100
            done = True

        # update snake coordinates
        self.snake.insert_head((i, j))
        self.map[i][j] = self.SNAKE
        if not grow:
            i, j = self.snake.pop_tail()
            self.map[i][j] = 0

        return self._get_state(), reward, done, {'step_length': self.step_len,
                                                 'snake_length': self.snake.locations.__len__(),
                                                 'snake_health': self.snake.health,
                                                 'eat_cnt': self.eat_cnt,
                                                 }
    
    def _get_state(self):
        # 3 channels, 1 channel for each snake, food, wall
        state = np.zeros((3, self.map_size, self.map_size), dtype=np.int32)
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[i][j] == self.SNAKE:
                    state[0][i][j] = 1
                elif self.map[i][j] == self.FOOD:
                    state[1][i][j] = 1
                elif self.map[i][j] == self.WALL:
                    state[2][i][j] = 1
        
        # return self.map.copy().flatten()
        return state.copy()
    
    def render(self, delay: float = 0.0):
        _ = os.system('cls' if os.name == 'nt' else 'clear')

        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[i][j] == 0:  # empty
                    print(' ', end=' ')
                elif self.map[i][j] == 1:  # snake
                    print('O', end=' ')
                elif self.map[i][j] == 2:  # food
                    print('*', end=' ')
                if self.map[i][j] == 3:  # obstacle
                    print('#', end=' ')
            print()
        
        if delay:
            time.sleep(delay)

    def _char2idx(self, char):
        if char == 'w':
            return self.UP
        elif char == 'd':
            return self.RIGHT
        elif char == 's':
            return self.DOWN
        elif char == 'a':
            return self.LEFT
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
