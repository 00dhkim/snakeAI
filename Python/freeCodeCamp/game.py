import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from typing import List

pygame.init()
font = pygame.font.Font('Python\\freeCodeCamp\\arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

BLOCK_SIZE = 20
SPEED = 1000 # 숫자가 클수록 빠름. 100이면 관찰하기 적당.

REWARD_GAME_OVER = -10
REWARD_EAT_FOOD = 10
REWARD_JUST_MOVE = -0.001

N_OBSTACLES = 0

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.direction = Direction.RIGHT
        self.head = Point(0, 0)
        self.snakes = []
        self.score = 0
        self.food = Point(0, 0)
        self.obstacles = []
        self.frame_iteration = 0
        self.episode = -1

        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snakes = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        self.score = 0
        self._place_food()
        self._generate_obstacle(N_OBSTACLES)
        self.frame_iteration = 0
        self.episode += 1

    def _place_food(self):
        while True:
            self.food = self._generate_random_point(self.w, self.h)
            if self.food not in self.snakes and self.food not in self.obstacles:
                break

    def play_step(self, action: List[int]):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snakes.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        # TODO: 무한루프에 빠질 수 있음
        if self.is_collision() or self.frame_iteration > 100 * len(self.snakes):
            game_over = True
            reward = REWARD_GAME_OVER
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = REWARD_EAT_FOOD
            self._place_food()
        else:
            reward = REWARD_JUST_MOVE
            self.snakes.pop()

        # 5. update ui and clock
        self._update_ui()
        global SPEED #FIXME: 디버그용
        self.clock.tick(SPEED)
        if self.episode > 200: #FIXME: 디버그용
            SPEED = 100

        # 6. return
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snakes[1:]:
            return True
        # hits obstacles
        if pt in self.obstacles:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # draw snakes
        for pt in self.snakes:
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        
        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw obstacles
        for pt in self.obstacles:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw head arrow, 머리 방향을 가리키는 직선
        delta = self.direction_to_delta(self.direction)
        pygame.draw.line(self.display, WHITE, 
                         start_pos=(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE / 2),
                         end_pos=(self.head.x + BLOCK_SIZE / 2 + delta.x, 
                                  self.head.y + BLOCK_SIZE / 2 + delta.y),
                         width=2)
        
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: List[int]):
        # [직진, 우회전, 좌회전]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _generate_obstacle(self, num_obstacle=1):
        '''
        랜덤 위치에 장애물을 생성
        생성 시 snake, food와 겹치지 않도록 함
        매 episode마다 호출됨
        '''
        self.obstacles: List[Point] = []
        for _ in range(num_obstacle):
            while True:
                obstacle = self._generate_random_point(self.w, self.h)
                if obstacle not in self.snakes \
                    and obstacle != self.food \
                    and obstacle not in self.obstacles:
                    
                    self.obstacles.append(obstacle)
                    break
    
    def _generate_random_point(self, w, h) -> Point:
        '''
        랜덤하게 Point 생성 후 리턴
        포인트 자리가 비었을지 보장하지 않음. (직접 체크해야 함)
        '''
        x = random.randint(0, (w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return Point(x, y)
    
    def direction_to_delta(self, direction: Direction) -> Point:
        '''
        direction을 받아서 delta를 리턴
        '''
        if direction == Direction.RIGHT:
            return Point(BLOCK_SIZE, 0)
        elif direction == Direction.LEFT:
            return Point(-BLOCK_SIZE, 0)
        elif direction == Direction.DOWN:
            return Point(0, BLOCK_SIZE)
        elif direction == Direction.UP:
            return Point(0, -BLOCK_SIZE)