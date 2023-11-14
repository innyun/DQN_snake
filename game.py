import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()

font = pygame.font.SysFont('arial', 24)

block_size = 20
speed = 40
width = 400
height = 400

render = False


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 150, 255)
BLACK = (0, 0, 0)


class SnakeGameAI:
    def __init__(self, w=width, h=height, speed=speed, render=False):
        self.w = w
        self.h = h
        self.speed = speed
        self.render = render

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def place_food(self):
        x = random.randint(0, (self.w - block_size) // block_size) * block_size
        y = random.randint(0, (self.h - block_size) // block_size) * block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def step(self, action):
        reward = -.01

        self.frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)
        self.snake.insert(0, self.head)

        game_over = False
        if self.is_collision() or self.frame > 100*len(self.snake):
            game_over = True
            reward -= max(10.0, math.sqrt(len(self.snake)))
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        if self.render:
            self.update_ui()
        self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x > self.w - block_size or pt.x < 0 or pt.y > self.h - block_size or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def update_ui(self):
        self.display.fill(BLACK)

        for i in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(i.x, i.y, block_size, block_size))

        pygame.draw.circle(self.display, RED, (self.food.x + (block_size / 2) + 1, self.food.y + (block_size / 2) + 1), block_size / 2 - .5)

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        i = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[i]
        elif np.array_equal(action, [0, 1, 0]):
            next_i = (i + 1) % 4
            new_dir = clock_wise[next_i]
        else:
            next_i = (i - 1) % 4
            new_dir = clock_wise[next_i]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size
        elif self.direction == Direction.DOWN:
            y += block_size
        elif self.direction == Direction.UP:
            y -= block_size

        self.head = Point(x, y)

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x - block_size, self.head.y), Point(self.head.x - (2 * block_size), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        self.frame = 0


# if __name__ == '__main__':
#     game = SnakeGameAI()
#
#     while True:
#         game_over, score = game.step()
#
#         if game_over:
#             break
#
#     print(f"Final score: {score}")
#
#     pygame.quit()
