import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot_helper import plot
from pygame_screen_record.ScreenRecorder import ScreenRecorder
import os

max_memory = 100_000
batch_size = 1000
lr = 0.0005

train_games = 1000
test_games = 100

train_speed = 10000
test_speed = 40


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = .9
        self.memory = deque(maxlen=max_memory)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > batch_size:
            sample = random.sample(self.memory, batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*sample)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # self.epsilon = max(.001, (80 - self.n_games) / 160)
        move = [0, 0, 0]
        self.epsilon = (250 - self.n_games) / 500
        if random.random() < self.epsilon:
            rand = random.randint(0, 2)
            move[rand] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            best = torch.argmax(prediction).item()
            move[best] = 1

        return move


def train():
    scores = []
    mean_scores = []
    total = 0
    record = 0

    game = SnakeGameAI(speed=train_speed, render=False)

    while agent.n_games < train_games:
        state_old = agent.get_state(game)

        move = agent.get_action(state_old)

        reward, game_over, score = game.step(move)
        state_new = agent.get_state(game)

        agent.train_memory(state_old, move, reward, state_new, game_over)

        agent.remember(state_old, move, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Training Phase: Game {agent.n_games}/{train_games}, Score {score}, Record {record}")

            scores.append(score)
            total += score
            mean_score = total / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)


def test():
    game = SnakeGameAI(speed=test_speed, render=True)
    record = 0

    recorder = ScreenRecorder(60).start_rec()

    i = 0

    while i < test_games:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, game_over, score = game.step(move)

        if game_over:
            recorder.stop_rec()
            game.reset()
            i += 1

            if score > record:
                if os.path.exists("./model/best_model_recording.mp4"):
                    os.remove("./model/best_model_recording.mp4")
                recording = recorder.get_single_recording()
                recording.save(("./model/best_model_recording", "mp4"))
                record = score

            print(f"Testing Phase: Game {i + 1}/{test_games}, Score {score}, Record {record}")

            recorder = ScreenRecorder(60).start_rec()

if __name__ == '__main__':
    agent = Agent()
    train()
    test()
