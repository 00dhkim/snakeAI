"""
snake.py

snake 게임의 기능을 담은 소스코드

"""

import random

MAP_SIZE = 50


class SnakeGame:
    def __init__(self):
        food_pos = []  # 먹이의 좌표
        snake_pos = []  # 뱀의 좌표
        wall_pos = []  # 벽의 좌표

        snake_length = 0

    def game_init(self):
        """
        게임 시작 시 초기 세팅

        :return:
        """

    def set_food(self):
        """
        매 턴마다 랜덤 위치에 먹이를 둠

        :return:
        """

    def move_able(self, user_input):
        """
        움직이게 될 자리에 뭐가 있는지 알고, 액션을 수행
        - 벽이 있다면, 사망
        - 먹이가 있다면, 성장

        :param user_input:
        :return:
        """

    def move(self):
        """
        move_able()에서 액션 후, 뱀을 이동시킴

        :return:
        """
