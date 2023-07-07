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
    direction을 받아서 진행 방향을 의미하는 Point 방향벡터를 리턴.
    벡터의 크기는 1
    '''
    if direction == Direction.RIGHT:
        return Point(1, 0)
    elif direction == Direction.LEFT:
        return Point(-1, 0)
    elif direction == Direction.DOWN:
        return Point(0, 1)
    elif direction == Direction.UP:
        return Point(0, -1)
