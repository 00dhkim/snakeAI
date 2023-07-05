import pygame

from typing import List
from helper import Direction, Point
from helper import generate_random_point, direction_converter
from helper import UI

BLOCK_SIZE = 20

REWARD_GAME_OVER = -10
REWARD_EAT_FOOD = 10
REWARD_JUST_MOVE = -0.001

N_OBSTACLES = 0

class SnakeGame:

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
        self.ui = UI(w, h)
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
            self.food = generate_random_point(self.w, self.h)
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
        self.ui.update(snakes=self.snakes, head=self.head, food=self.food, obstacles=self.obstacles, direction=self.direction, score=self.score)

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

    def _move(self, action: List[int]):
        
        self.direction = direction_converter(self.direction, action)

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
                obstacle = generate_random_point(self.w, self.h)
                if obstacle not in self.snakes \
                    and obstacle != self.food \
                    and obstacle not in self.obstacles:
                    
                    self.obstacles.append(obstacle)
                    break



