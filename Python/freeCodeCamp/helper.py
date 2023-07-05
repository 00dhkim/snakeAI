import matplotlib.pyplot as plt
from IPython import display

import random
import numpy as np

from enum import Enum
from typing import List
from collections import namedtuple

import pygame

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20

ACTION_STRAIGHT = [1, 0, 0]
ACTION_RIGHT = [0, 1, 0]
ACTION_LEFT = [0, 0, 1]

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

SPEED = 10000 # 숫자가 클수록 빠름. 100이면 관찰하기 적당.

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    # plt.axhline(y=mean_scores[-1], color='r', linestyle='-') # 마지막 mean_score에 대한 수평선
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def savefig(filename):
    plt.title('Train Result')
    plt.savefig(filename)

def direction_converter(direction: Direction, action: List[int]) -> Direction:
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(direction)
    
    if np.array_equal(action, ACTION_STRAIGHT):
        new_dir = clock_wise[idx]  # no change
    elif np.array_equal(action, ACTION_RIGHT):
        next_idx = (idx + 1) % 4
        new_dir = clock_wise[next_idx]  # right turn
    else:  # [0, 0, 1]
        next_idx = (idx - 1) % 4
        new_dir = clock_wise[next_idx]  # left turn
    
    return new_dir

def generate_random_point(w, h) -> Point:
    '''
    랜덤하게 Point 생성 후 리턴
    포인트 자리가 비었을지 보장하지 않음. (직접 체크해야 함)
    '''
    x = random.randint(0, (w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    y = random.randint(0, (h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    return Point(x, y)

def direction_to_delta(direction: Direction) -> Point:
    '''
    direction을 받아서 진행 방향을 의미하는 Point 방향벡터를 리턴
    '''
    if direction == Direction.RIGHT:
        return Point(BLOCK_SIZE, 0)
    elif direction == Direction.LEFT:
        return Point(-BLOCK_SIZE, 0)
    elif direction == Direction.DOWN:
        return Point(0, BLOCK_SIZE)
    elif direction == Direction.UP:
        return Point(0, -BLOCK_SIZE)

# 직선을 그리려면 agent.py의 state에 있는 dist_straight, dist_right, dist_left가 필요함.
# 즉, agent.py의 Agent 클래스와 game.py의 SnakeGameAI 클래스 모두에 의존해야 함.
# 그런데 game.py에서는 ui.py의 UI 클래스의 update_ui 함수를 호출해야 함.
class UI:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        pygame.init()
        self.font = pygame.font.SysFont('arial', 25)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
    
    def update(self, snakes, head, food, obstacles, direction, score):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self.display.fill(BLACK)
        
        # draw snakes
        for pt in snakes:
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw obstacles
        for pt in obstacles:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw head arrow, 머리 방향을 가리키는 직선
        delta = direction_to_delta(direction)
        pygame.draw.line(self.display, YELLOW, 
                         start_pos=(head.x + BLOCK_SIZE / 2, head.y + BLOCK_SIZE / 2),
                         end_pos=(head.x + BLOCK_SIZE / 2 + delta.x, 
                                  head.y + BLOCK_SIZE / 2 + delta.y),
                         width=2)
        
        text = self.font.render(f"Score: {score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(SPEED)