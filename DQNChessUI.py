# DQNChessUI.py

import pygame
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- Config ---
WIDTH, HEIGHT = 480, 480
SQ_SIZE = WIDTH // 8
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)

STATE_SIZE = 64
HIDDEN_SIZE = 128
ACTION_SIZE = 4672
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.001
BATCH_SIZE = 64

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# --- Agent ---
class DQNAgent:
    def __init__(self):
        self.model = DQN(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
        self.target_model = DQN(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
        self.memory = deque(maxlen=2000)
        self.epsilon = EPSILON_START
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(legal_moves)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state_tensor).detach().numpy()
        legal_q = [(move, q_values[move_to_index(move)]) for move in legal_moves]
        best_move = max(legal_q, key=lambda x: x[1])[0]
        return best_move

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            target = self.model(state_tensor).clone().detach()
            if done:
                target[move_to_index(action)] = reward
            else:
                next_q = self.target_model(next_state_tensor).detach()
                target[move_to_index(action)] = reward + GAMMA * torch.max(next_q)
            output = self.model(state_tensor)
            loss = nn.MSELoss()(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


# --- Utility ---
def board_to_state(board):
    state = np.zeros(64)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            state[i] = {
                'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
            }[piece.symbol()]
    return state


def move_to_index(move):
    return move.from_square * 73 + move.to_square % 73


def draw_board(screen, board, images):
    for r in range(8):
        for c in range(8):
            color = LIGHT_BROWN if (r + c) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            screen.blit(images[piece.symbol()], (col * SQ_SIZE, row * SQ_SIZE))


def load_piece_images():
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    images = {}
    for p in pieces:
        img = pygame.image.load(f"pieces/{'w' if p.isupper() else 'b'}{p.upper()}.png")
        images[p] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
    return images


# --- Main ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DQN Chess AI")
images = load_piece_images()
clock = pygame.time.Clock()

board = chess.Board()
agent = DQNAgent()
running = True

while running and not board.is_game_over():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = board_to_state(board)
    legal_moves = list(board.legal_moves)
    move = agent.act(state, legal_moves)
    board.push(move)
    next_state = board_to_state(board)
    reward = 1 if board.is_checkmate() else 0
    agent.remember(state, move, reward, next_state, board.is_game_over())
    agent.replay()

    draw_board(screen, board, images)
    pygame.display.flip()
    clock.tick(1)

pygame.quit()
