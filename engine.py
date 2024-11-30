import torch
import torch.nn as nn
import random
import chess
import os
from math import sqrt, log


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, 64)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # 가치 평가는 -1 ~ 1로 제한
        return policy, value


class TreeNode:
    def __init__(self, board, parent=None, prior_prob=1.0):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.prior_prob = prior_prob

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, exploration_param=1.4):
        return max(
            self.children.values(),
            key=lambda child: (child.wins / child.visits) + exploration_param * sqrt(log(self.visits) / child.visits)
        )

    def update(self, result):
        self.visits += 1
        self.wins += result


def fen_to_tensor(fen):
    board = chess.Board(fen)
    board_array = []
    
    # 각 칸을 숫자로 변환 (빈 칸은 0, 백색 기물은 1~6, 흑색 기물은 -1~-6)
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            board_array.append(0)
        else:
            board_array.append(piece.piece_type if piece.color == chess.WHITE else -piece.piece_type)
    
    return torch.tensor(board_array, dtype=torch.float32)



def select(node):
    while not node.board.is_game_over() and node.is_fully_expanded():
        node = node.best_child()
    return node

def expand(node, policy_net):
    legal_moves = list(node.board.legal_moves)
    move = random.choice(legal_moves)
    new_board = node.board.copy()
    new_board.push(move)
    
    board_tensor = fen_to_tensor(new_board.board_fen())

    policy, _ = policy_net(board_tensor)
    policy = torch.softmax(policy, dim=0)

    move_index = legal_moves.index(move)
    prior_prob = policy[move_index].item()
    
    child_node = TreeNode(new_board, parent=node, prior_prob=prior_prob)
    node.children[move] = child_node
    return child_node

def simulate(board):
    sim_board = board.copy()
    
    while not sim_board.is_game_over():
        move = random.choice(list(sim_board.legal_moves))
        sim_board.push(move)

    result = sim_board.result()

    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    
    return 0

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.wins += result
        result = -result
        node = node.parent


def train_step(policy_net, optimizer, board_state, policy, reward):
    board_tensor = fen_to_tensor(board_state)

    pred_policy, pred_value = policy_net(board_tensor)
    value_loss = nn.MSELoss()(pred_value, torch.tensor([reward]).float())
    policy_loss = -(policy * pred_policy.log()).mean()
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def load_model(model, path="chess_ai.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))