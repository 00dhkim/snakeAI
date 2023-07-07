import pygame
import numpy as np

from typing import List
from helper import Direction, Point
from helper import generate_random_point, direction_converter, direction_to_delta

pygame.init()
font = pygame.font.SysFont('arial', 25)

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

BLOCK_SIZE = 20
SPEED = 10000 # 숫자가 클수록 빠름. 100이면 관찰하기 적당.

REWARD_GAME_OVER = -10
REWARD_EAT_FOOD = 10
REWARD_JUST_MOVE = -0.001

N_ROCKS = 0

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.direction = Direction.RIGHT
        self.head = Point(0, 0)
        self.snakes = []
        self.score = 0
        self.food = Point(0, 0)
        self.rocks = []
        self.frame_iteration = 0
        self.episode = -1

        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.ui_speed = SPEED
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
        self._generate_rock(N_ROCKS)
        self.frame_iteration = 0
        self.episode += 1

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
        if self._is_collision() or self.frame_iteration > 100 * len(self.snakes):
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
        self.clock.tick(self.ui_speed)
        if self.episode > 200:
            self.ui_speed = 100
        

        # 6. return
        return reward, game_over, self.score

    def get_state(self):
        head = self.snakes[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = True if self.direction == Direction.LEFT else False
        dir_r = True if self.direction == Direction.RIGHT else False
        dir_u = True if self.direction == Direction.UP else False
        dir_d = True if self.direction == Direction.DOWN else False

        # straight-line obstacle and snake distance
        dist_straight = None
        for obs in self.obstacles + self.snakes[1:]:
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
                dist_straight = self.w - head.x
            elif dir_u:
                dist_straight = head.y
            elif dir_d:
                dist_straight = self.h - head.y
            else:
                assert False, "direction error"
        
        # right-turn obstacle and snake distance
        dist_right = None
        for obs in self.obstacles + self.snakes[1:]:
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
                dist_right = self.h - head.y
            elif dir_r:
                dist_right = head.y
            elif dir_u:
                dist_right = head.x
            elif dir_d:
                dist_right = self.w - head.x
            else:
                assert False, "direction error"
        
        # left-turn obstacle and snake distance
        dist_left = None
        for obs in self.obstacles + self.snakes[1:]:
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
                dist_left = self.h - head.y
            elif dir_u:
                dist_left = self.w - head.x
            elif dir_d:
                dist_left = head.x
            else:
                assert False, "direction error"
        
        # normalize distance
        if dir_l or dir_r:
            dist_straight = dist_straight / self.w / BLOCK_SIZE
            dist_right = dist_right / self.h / BLOCK_SIZE
            dist_left = dist_left / self.h / BLOCK_SIZE
        elif dir_u or dir_d:
            dist_straight = dist_straight / self.h / BLOCK_SIZE
            dist_right = dist_right / self.w / BLOCK_SIZE
            dist_left = dist_left / self.w / BLOCK_SIZE
        else:
            assert False, "direction error"
        
        state = [
            # state[0] danger straight
            dir_r and self._is_collision(point_r) or \
            dir_l and self._is_collision(point_l) or \
            dir_u and self._is_collision(point_u) or \
            dir_d and self._is_collision(point_d),  # 아래로 가고있는데 아래쪽에 벽이있다? 직진 위험.

            # state[1] danger right
            dir_u and self._is_collision(point_r) or \
            dir_d and self._is_collision(point_l) or \
            dir_l and self._is_collision(point_u) or \
            dir_r and self._is_collision(point_d),  # 오른쪽으로 가고있는데 아래쪽에 벽이있다? 우회전 위험.

            # state[2] danger left
            dir_d and self._is_collision(point_r) or \
            dir_u and self._is_collision(point_l) or \
            dir_r and self._is_collision(point_u) or \
            dir_l and self._is_collision(point_d),  # 왼쪽으로 가고있는데 아래쪽에 벽이있다? 좌회전 위험.

            # state[3:6] 진행방향에서 head와 장애물간의 거리(정규화되어있음, 0~1)
            dist_straight,
            dist_right,
            dist_left,

            # state[6:10] move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # state[10:14] food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snakes[1:]:
            return True
        # hits rock
        if pt in self.rocks:
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
        
        # draw rocks
        for pt in self.rocks:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # draw head arrow, 머리 방향을 가리키는 직선
        delta = direction_to_delta(self.direction)
        pygame.draw.line(self.display, WHITE, 
                         start_pos=(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE / 2),
                         end_pos=(self.head.x + BLOCK_SIZE / 2 + delta.x, 
                                  self.head.y + BLOCK_SIZE / 2 + delta.y),
                         width=2)
        
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _place_food(self):
        while True:
            self.food = generate_random_point(self.w, self.h)
            if self.food not in self.snakes and self.food not in self.rocks:
                break

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
        
    def _generate_rocks(self, num_rocks=1):
        '''
        랜덤 위치에 장애물을 생성
        생성 시 snake, food와 겹치지 않도록 함
        매 episode마다 호출됨
        '''
        self.obstacles: List[Point] = []
        for _ in range(num_rocks):
            while True:
                rock = generate_random_point(self.w, self.h)
                if rock not in self.snakes \
                    and rock != self.food \
                    and rock not in self.rocks:
                    
                    self.rocks.append(rock)
                    break



