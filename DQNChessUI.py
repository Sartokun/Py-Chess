import pygame
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

# --- Config ---
WIDTH, HEIGHT = 480, 480
SQ_SIZE = WIDTH // 8
SIDE_BAR_WIDTH = 300
STATS_BAR_WIDTH = 200
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
UI_BG_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)

STATE_SIZE = 64
HIDDEN_SIZE = 128
ACTION_SIZE = 4672
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.001
BATCH_SIZE = 64

num_self_play_games = 20

stats = {
    "Wins": 0,
    "Moves": 0,
    "Pieces Taken": 0,
    "Q Accuracy": [],
}

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
    return move.from_square * 64 + move.to_square


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


def draw_ui(screen, game_count, move_history, scroll_offset, images):
    # แถบด้านข้าง UI
    pygame.draw.rect(screen, UI_BG_COLOR, pygame.Rect(WIDTH, 0, SIDE_BAR_WIDTH, HEIGHT))

    # ข้อความจำนวนเกมที่เล่นไปแล้ว
    font = pygame.font.SysFont('Arial', 24)
    text = font.render(f"Games Played: {game_count}", True, TEXT_COLOR)
    screen.blit(text, (WIDTH + 20, 20))

    # ประวัติการเดิน
    text = font.render("Move History:", True, TEXT_COLOR)
    screen.blit(text, (WIDTH + 20, 60))
    y_offset = 100 - scroll_offset  # ปรับค่าการเลื่อนขึ้น-ลง
    for move in move_history[-10:]:  # แสดงเฉพาะ 10 การเดินล่าสุด
        piece = board.piece_at(move.to_square)

        # ตรวจสอบว่า piece ไม่เป็น None
        if piece:
            # ตรวจสอบสีของหมาก
            color = 'White' if piece.color == chess.WHITE else 'Black'

            # แปลงตำแหน่งจาก square เป็นชื่อช่อง (เช่น E4 -> E5)
            from_square_name = chess.square_name(move.from_square)
            to_square_name = chess.square_name(move.to_square)
            
            # แสดงไอคอนของหมากที่ย้ายไปที่ช่อง `to_square`
            piece_icon = pygame.transform.scale(images[piece.symbol()], (30, 30))  # ปรับขนาดไอคอนเป็น 30x30 พิกเซล
            screen.blit(piece_icon, (WIDTH + 10, y_offset))
            
            # ข้อความการเดิน
            move_text = f"{color} {piece.symbol().upper()} ({from_square_name} -> {to_square_name})"
            text = font.render(move_text, True, TEXT_COLOR)
            screen.blit(text, (WIDTH + 50, y_offset))
            
        else:
            from_square_name = chess.square_name(move.from_square)
            to_square_name = chess.square_name(move.to_square)
            move_text = f"Invalid Move ({from_square_name} -> {to_square_name})"
            text = font.render(move_text, True, TEXT_COLOR)
            screen.blit(text, (WIDTH + 20, y_offset))

        y_offset += 30

    # แถบเลื่อน
    scrollbar_height = 300
    scrollbar_pos = pygame.Rect(WIDTH + SIDE_BAR_WIDTH - 20, 100, 10, scrollbar_height)
    pygame.draw.rect(screen, (100, 100, 100), scrollbar_pos)

    # ปรับความยาวแถบเลื่อนให้ยาวตามจำนวนการเดิน
    total_moves = len(move_history)
    max_scroll_offset = max(0, total_moves - 10)
    scroll_size = min(30, scrollbar_height * 10 / total_moves if total_moves > 10 else scrollbar_height)
    scroll_pos = (scroll_offset / max_scroll_offset) * (scrollbar_height - scroll_size) if max_scroll_offset > 0 else 0
    pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(WIDTH + SIDE_BAR_WIDTH - 20, 100 + scroll_pos, 10, scroll_size))
    
    
# --- UI Stats ---  
def draw_stats_ui(screen, stats):
    # แถบด้านข้าง UI
    pygame.draw.rect(screen, UI_BG_COLOR, pygame.Rect(WIDTH + SIDE_BAR_WIDTH, 0, STATS_BAR_WIDTH, HEIGHT))
    font = pygame.font.SysFont('Arial', 22)
    y = 20
    screen.blit(font.render("AI Stats", True, TEXT_COLOR), (WIDTH + SIDE_BAR_WIDTH, y))
    y += 40
    for key, value in stats.items():
        if isinstance(value, list):
            avg = sum(value) / len(value) if value else 0
            text = f"{key}: {avg:.2f}"
        else:
            text = f"{key}: {value}"
        screen.blit(font.render(text, True, TEXT_COLOR), (WIDTH + SIDE_BAR_WIDTH, y))
        y += 30


def play_game():
    running = True
    while running and not board.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        state = board_to_state(board)
        legal_moves = list(board.legal_moves)
        move = agent.act(state, legal_moves)  # AI ทำการเดิน
        board.push(move)
        next_state = board_to_state(board)
        reward = 1 if board.is_checkmate() else 0
        agent.remember(state, move, reward, next_state, board.is_game_over())
        agent.replay()

        draw_board(screen, board, images)
        draw_ui(screen, game_count, move_history, scroll_offset, images)
        draw_stats_ui(screen, stats)

        pygame.display.flip()
        clock.tick(0)  # เพิ่มการหน่วงเวลาหรือปรับให้เหมาะสม

        move_history.append(move)
        stats["Moves"] += 1
        if board.is_capture(move):
            stats["Pieces Taken"] += 1

        
# --- Main ---
pygame.init()
screen = pygame.display.set_mode((WIDTH + SIDE_BAR_WIDTH + STATS_BAR_WIDTH, HEIGHT))
pygame.display.set_caption("DQN Chess AI")
images = load_piece_images()
clock = pygame.time.Clock()

board = chess.Board()
agent = DQNAgent()

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

game_count = 0
move_history = []
scroll_offset = 0

for game_num in range(num_self_play_games):
    board = chess.Board()
    
    agent.model.load_state_dict(torch.load('best_model.pth'))
    agent.model.eval()

    while not board.is_game_over():
        play_game()

        print(f"Game {game_num + 1} finished.")
        game_count += 1

        torch.save(agent.model.state_dict(), os.path.join(model_dir, f"model_game_{game_num + 1}.pth"))
        torch.save(agent.model.state_dict(), 'best_model.pth')
        
        if board.result() == '1-0':
            stats["Wins"] += 1

        # ประเมินความแม่นยำจาก Q-value
        state_tensor = torch.tensor(board_to_state(board), dtype=torch.float32)
        q_values = agent.model(state_tensor).detach().numpy()
        legal_indices = [move_to_index(m) for m in board.legal_moves]
        legal_q_values = [q_values[i] for i in legal_indices]
        
        if legal_q_values:
            stats["Q Accuracy"].append(max(legal_q_values))
        else:
            stats["Q Accuracy"].append(0)
        
        print(f"----- stats Game : {game_num + 1} -----")
        print(stats)


torch.save(agent.model.state_dict(), 'best_model.pth')

pygame.quit()
print("----- stats finished -----")
print(stats)