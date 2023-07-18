import pygame
import numpy as np

from typing import List, Tuple
from helper import Direction, Point
from helper import generate_random_point, direction_converter, direction_to_delta

pygame.init()
font = pygame.font.SysFont('arial', 25)
fontSmall = pygame.font.SysFont('arial', 12)

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

REWARD_GAME_OVER = -1
REWARD_EAT_FOOD = 1
REWARD_JUST_MOVE = -0.001
# REWARD_JUST_MOVE = 0

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
        self._generate_rocks(N_ROCKS)
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
        if self._is_collision() or self.frame_iteration > 100 * len(self.snakes):
            game_over = True
            reward = REWARD_GAME_OVER
            self._update_ui_gameover()
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
        # if self.episode > 200:
            # self.ui_speed = 100
        

        # 6. return
        return reward, game_over, self.score

    def get_state(self) -> np.ndarray:
        head = self.snakes[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = True if self.direction == Direction.LEFT else False
        dir_r = True if self.direction == Direction.RIGHT else False
        dir_u = True if self.direction == Direction.UP else False
        dir_d = True if self.direction == Direction.DOWN else False
        
        dist_straight, dist_right, dist_left = self._get_obstacle_distance()
        
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

            # state[3:7] move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # state[7:11] food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]
        
        window = self._get_window(self.head, window_size=5) # length: 50
        # window = np.array(window, dtype=np.float32) / 4 # window 정규화할 필요 없음 (2bit로 표현되어있음)

        return np.concatenate((state, window))

    def _get_obstacle_distance(self) -> Tuple[int, int, int]:
        '''
        head와 직진, 우회전, 좌회전 방향에 있는 장애물과의 거리를 구한다.
        
        reutrns: (dist_straight, dist_right, dist_left)
        '''
        head = self.snakes[0]

        dir_l = True if self.direction == Direction.LEFT else False
        dir_r = True if self.direction == Direction.RIGHT else False
        dir_u = True if self.direction == Direction.UP else False
        dir_d = True if self.direction == Direction.DOWN else False  
        
        # straight-line rocks and snake distance
        dist_straight = 99999
        for obs in self.rocks + self.snakes[1:]:
            if dir_l:
                # obs 중에서 head보다 왼쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x < head.x and obs.y == head.y and dist_straight > head.x - obs.x:
                    dist_straight = head.x - obs.x
            elif dir_r:
                # obs 중에서 head보다 오른쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x > head.x and obs.y == head.y and dist_straight > obs.x - head.x:
                    dist_straight = obs.x - head.x
            elif dir_u:
                # obs 중에서 head보다 위쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y < head.y and dist_straight > head.y - obs.y:
                    dist_straight = head.y - obs.y
            elif dir_d:
                # obs 중에서 head보다 아래쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y > head.y and dist_straight > obs.y - head.y:
                    dist_straight = obs.y - head.y
        if dist_straight  == 99999:
            # head와 벽 끝까지의 거리
            if dir_l:
                dist_straight = head.x + BLOCK_SIZE
            elif dir_r:
                dist_straight = self.w - head.x
            elif dir_u:
                dist_straight = head.y + BLOCK_SIZE
            elif dir_d:
                dist_straight = self.h - head.y
            else:
                assert False, "direction error"
        
        # right-turn rocks and snake distance
        dist_right = 99999
        for obs in self.rocks + self.snakes[1:]:
            if dir_l:
                # obs 중에서 head보다 위쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y < head.y and dist_right > head.y - obs.y:
                    dist_right = head.y - obs.y
            elif dir_r:
                # obs 중에서 head보다 아래쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y > head.y and dist_right > obs.y - head.y:
                    dist_right = obs.y - head.y
            elif dir_u:
                # obs 중에서 head보다 오른쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x > head.x and obs.y == head.y and dist_right > obs.x - head.x:
                    dist_right = obs.x - head.x
            elif dir_d:
                # obs 중에서 head보다 왼쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x < head.x and obs.y == head.y and dist_right > head.x - obs.x:
                    dist_right = head.x - obs.x
        if dist_right == 99999:
            # head와 벽 끝까지의 거리
            if dir_l:
                dist_right = head.y + BLOCK_SIZE
            elif dir_r:
                dist_right = self.h - head.y
            elif dir_u:
                dist_right = self.w - head.x
            elif dir_d:
                dist_right = head.x + BLOCK_SIZE
            else:
                assert False, "direction error"
        
        # left-turn rocks and snake distance
        dist_left = 99999
        for obs in self.rocks + self.snakes[1:]:
            if dir_l:
                # obs 중에서 head보다 아래쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y > head.y and dist_left > obs.y - head.y:
                    dist_left = obs.y - head.y
            elif dir_r:
                # obs 중에서 head보다 위쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x == head.x and obs.y < head.y and dist_left > head.y - obs.y:
                    dist_left = head.y - obs.y
            elif dir_u:
                # obs 중에서 head보다 왼쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x < head.x and obs.y == head.y and dist_left > head.x - obs.x:
                    dist_left = head.x - obs.x
            elif dir_d:
                # obs 중에서 head보다 오른쪽에 있는 것들 중에서 가장 가까운 것
                if obs.x > head.x and obs.y == head.y and dist_left > obs.x - head.x:
                    dist_left = obs.x - head.x
        if dist_left == 99999:
            # head와 벽 끝까지의 거리
            if dir_l:
                dist_left = self.h - head.y
            elif dir_r:
                dist_left = head.y + BLOCK_SIZE
            elif dir_u:
                dist_left = head.x + BLOCK_SIZE
            elif dir_d:
                dist_left = self.w - head.x
            else:
                assert False, "direction error"
        
        return dist_straight, dist_right, dist_left

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return 1
        # hits itself
        if pt in self.snakes[1:]:
            return 2
        # hits rock
        if pt in self.rocks:
            return 3

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
        # pygame.draw.line(self.display, WHITE, 
        #                  start_pos=(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE / 2),
        #                  end_pos=(self.head.x + BLOCK_SIZE / 2 + delta.x * BLOCK_SIZE, 
        #                           self.head.y + BLOCK_SIZE / 2 + delta.y * BLOCK_SIZE),
        #                  width=2)
        
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
        delta = direction_to_delta(self.direction)
        dist_straight, dist_right, dist_left = self._get_obstacle_distance()
        dist_straight -= BLOCK_SIZE
        dist_right -= BLOCK_SIZE
        dist_left -= BLOCK_SIZE
        
        # # draw straight line, 머리부터 직진방향으로 장애물(맵 끝, 돌, 뱀)까지의 직선
        # pygame.draw.line(self.display, (255,128,128),
        #                  start_pos=(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE / 2),
        #                  end_pos=(self.head.x + BLOCK_SIZE / 2 + delta.x * dist_straight,
        #                           self.head.y + BLOCK_SIZE / 2 + delta.y * dist_straight),
        #                  width=1)
        
        # # draw right line, 머리부터 우회전방향으로 장애물(맵 끝, 돌, 뱀)까지의 직선
        # pygame.draw.line(self.display, (128,255,128),
        #                  start_pos=(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE / 2),
        #                  end_pos=(self.head.x + BLOCK_SIZE / 2 - delta.y * dist_right,
        #                          self.head.y + BLOCK_SIZE / 2 + delta.x * dist_right),
        #                  width=1)
        
        # # draw left line, 머리부터 좌회전방향으로 장애물(맵 끝, 돌, 뱀)까지의 직선
        # pygame.draw.line(self.display, (128,128,255),
        #                  start_pos=(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE / 2),
        #                  end_pos=(self.head.x + BLOCK_SIZE / 2 + delta.y * dist_left,
        #                           self.head.y + BLOCK_SIZE / 2 - delta.x * dist_left),
        #                  width=1)
        
        # draw window, 5x5 크기의 흰색 박스 생성, 각 칸마다 window 리스트의 값을 텍스트로 표시
        pygame.draw.rect(self.display, WHITE, 
            pygame.Rect(self.head.x - 2 * BLOCK_SIZE, self.head.y - 2 * BLOCK_SIZE, 5 * BLOCK_SIZE, 5 * BLOCK_SIZE),
            width=1)
        window = self._get_window_str(self.head, window_size=5)
        for i in range(5):
            for j in range(5):
                text = fontSmall.render(f"{window[i*5+j]}", True, WHITE)
                self.display.blit(text, [self.head.x - 2 * BLOCK_SIZE + j * BLOCK_SIZE, self.head.y - 2 * BLOCK_SIZE + i * BLOCK_SIZE])
        
        pygame.display.flip()

    def _update_ui_gameover(self):
        '''
        게임오버 이유를 화면에 표시.
        다음 게임이 시작되기 전까지, 짧은 학습시간동안 화면에 나타난다
        '''
        r = self._is_collision()
        if r == 1:
            text = font.render(f"Game Over: Hit the Wall", True, WHITE)
        elif r == 2:
            text = font.render(f"Game Over: Hit Yourself", True, WHITE)
        elif r == 3:
            text = font.render(f"Game Over: Hit the Rock", True, WHITE)
        else:
            text = font.render(f"Game Over: Timeout", True, WHITE)
        self.display.blit(text, [self.w / 2 - 100, self.h / 2])
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
        self.rocks: List[Point] = []
        for _ in range(num_rocks):
            while True:
                rock = generate_random_point(self.w, self.h)
                if rock not in self.snakes \
                    and rock != self.food \
                    and rock not in self.rocks:
                    
                    self.rocks.append(rock)
                    break

    def _get_window_str(self, head: Point, window_size: int = 5) \
        -> List[str]:
        '''
        head를 중심으로 window_size만큼의 사각형 영역을 구하고,
        그 영역에 있는 snake, rock, food를 1차원 배열로 반환한다. (가로로 먼저, 마치 글 읽듯이)
        '''
        window = []
        for i in range(-window_size//2 + 1, window_size//2 + 1): # -2, -1, 0, 1, 2
            for j in range(-window_size//2 + 1, window_size//2 + 1): # -2, -1, 0, 1, 2
                pt = Point(head.x + j * BLOCK_SIZE, head.y + i * BLOCK_SIZE)
                if self._is_collision(pt):
                    window.append('o')
                elif pt == self.head:
                    window.append('o')
                elif pt == self.food:
                    window.append('f')
                else:
                    window.append('-')
        return window

    def _get_window(self, head: Point, window_size: int = 5) \
        -> List[int]:
        '''
        head를 중심으로 window_size만큼의 사각형 영역을 구하고,
        그 영역에 있는 snake, rock, food를 1차원 배열로 반환한다. (가로로 먼저, 마치 글 읽듯이)
        - [0, 0]: empty
        - [0, 1]: obstacle(boundary, snake, rock)
        - [1, 0]: food
        '''
        window = []
        for i in range(-window_size//2 + 1, window_size//2 + 1): # -2, -1, 0, 1, 2
            for j in range(-window_size//2 + 1, window_size//2 + 1): # -2, -1, 0, 1, 2
                pt = Point(head.x + j * BLOCK_SIZE, head.y + i * BLOCK_SIZE)
                if self._is_collision(pt):
                    window += [0, 1]
                elif pt == self.head:
                    window += [0, 1]
                elif pt == self.food:
                    window += [1, 0]
                else:
                    window += [0, 0]
        return window