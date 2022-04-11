"""
snake.py

snake 게임의 기능을 담은 소스코드

"""


import os
import random
import numpy as np
import time

MAP_SIZE = 10


class SnakeGame:
    def __init__(self):
        '''
        map label
        0: empty space
        1: snake
        2: food
        3: obstacle
        
        direction label
        0: up
        1: right
        2: down
        3: left
        
        reward
        -  +10: eat food
        - -100: dead
        -   -1: move
        '''
        self.state_size = MAP_SIZE * MAP_SIZE
        self.action_size = 4
        
        self.snake = None
        self.map = None
        

    
    def reset(self):
        random.seed(411)
        self.snake = [[0, 0]]
        self.map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
        
        for i in range(MAP_SIZE):
            self.map[0][i] = 3
            self.map[i][0] = 3
            self.map[MAP_SIZE - 1][i] = 3
            self.map[i][MAP_SIZE - 1] = 3
        
        self.snake[0] = [random.randint(1, MAP_SIZE - 2), random.randint(1, MAP_SIZE - 2)]
        self.map[self.snake[0][0]][self.snake[0][1]] = 1
        
        self._set_food()
        # self._set_obstacle()
        
        state = self.map.copy().flatten()
        return state

    def _set_food(self):
        """
        랜덤 위치에 먹이를 둠, 매 턴마다 수행
        """
        while True:
            i, j = random.randint(1, MAP_SIZE - 2), random.randint(1, MAP_SIZE - 2)
            if self.map[i][j] == 0:
                break
        self.map[i][j] = 2


    def _set_obstacle(self):
        """
        랜덤 위치에 장애물을 둠, 처음에만 수행
        """
        while True:
            i, j  = random.randint(1, MAP_SIZE - 2), random.randint(1, MAP_SIZE - 2)
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
        reward = 0
        reward = -1 # 움직이기만 해도 -1
        done = False
        grow = False
        
        if action == 0:   # up
            direc = [-1, 0]
        elif action == 1: # right
            direc = [0, 1]
        elif action == 2: # down
            direc = [1, 0]
        elif action == 3: # left
            direc = [0, -1]
        else:
            raise ValueError('input must be 0, 1, 2, 3')
        
        i, j = self.snake[0][0] + direc[0], self.snake[0][1] + direc[1]
        if self.map[i][j] == 2: # food
            grow = True
            reward += 10
            self._set_food()
        elif self.map[i][j] == 1 or self.map[i][j] == 3: # snake of obstacle
            reward -= 100
            done = True
        
        self.snake.insert(0, [i, j])
        self.map[i][j] = 1
        if not grow:
            i, j = self.snake.pop()
            self.map[i][j] = 0
        
        state = self.map.copy().flatten()
        return state, reward, done, {}


    def render(self):
        time.sleep(0.1)
        _ = os.system('cls' if os.name == 'nt' else 'clear')
        
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if self.map[i][j] == 0: # empty
                    print(' ', end=' ')
                elif self.map[i][j] == 1: # snake
                    print('O', end=' ')
                elif self.map[i][j] == 2: # food
                    print('*', end=' ')
                if self.map[i][j] == 3: # obstacle
                    print('#', end=' ')
            print()


    def _char2idx(self, char):
        if char == 'w':
            return 0
        elif char == 'd':
            return 1
        elif char == 's':
            return 2
        elif char == 'a':
            return 3
        else:
            print('input must be 0, 1, 2, 3')
            return 'invalidInput'


if __name__ == '__main__':
    game = SnakeGame()
    game.reset()
    
    while True:
        while True:
            print('input>', end='', flush=True)
            action = game._char2idx(input())
            if action != 'invalidInput':
                break
        
        state, reward, done, info = game.step(action)
        if done:
            game.reset()
        
        game.render()
