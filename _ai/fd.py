import chess
import chess.engine
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sqrt, log
import os


# 체스 정책 및 가치 네트워크 정의
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 64)  # 정책 헤드 (수 선택)
        self.value_head = nn.Linear(64, 1)    # 가치 헤드 (보드 평가)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # 가치 평가는 -1 ~ 1로 제한
        return policy, value

# 트리 노드 클래스 정의
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


# MCTS의 선택, 확장, 시뮬레이션, 역전파 단계 정의
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
        result = -result  # 상대 입장에서의 결과
        node = node.parent



# 모델 학습 단계 정의
def train_step(policy_net, optimizer, board_state, policy, reward):
    board_tensor = fen_to_tensor(board_state)

    pred_policy, pred_value = policy_net(board_tensor)
    value_loss = nn.MSELoss()(pred_value, torch.tensor([reward]).float())
    policy_loss = -(policy * pred_policy.log()).mean()
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 모델 저장 및 로드 함수
def save_model(model, path="chess_ai.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="chess_ai.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))



# 메인 자기 대국 루프
def main():
    policy_net = ChessNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    best_reward = float('-inf')

    for episode in range(100):
        print(f"Episode {episode}")
        board = chess.Board()
        root = TreeNode(board)
        total_reward = 0

        while not board.is_game_over():
            selected_node = select(root)

            if not selected_node.board.is_game_over():
                selected_node = expand(selected_node, policy_net)

            # 시뮬레이션을 통해 보드의 결과를 가져옵니다.
            reward = simulate(selected_node.board)
            backpropagate(selected_node, reward)

            # 모델 학습
            board_state = selected_node.board.fen()  # FEN 문자열로 가져옵니다.
            board_tensor = fen_to_tensor(board_state)

            # 정책 네트워크에 전달
            policy, _ = policy_net(board_tensor)  # board_tensor를 직접 사용
            train_step(policy_net, optimizer, board_state, policy, reward)

            # board를 업데이트
            board = selected_node.board.copy()  # selected_node의 보드 상태로 업데이트
            root = selected_node  # 루트 노드를 업데이트하여 다음 사이클에서 사용할 수 있도록 함
            total_reward += reward


        if total_reward > best_reward:
            best_reward = total_reward
            save_model(policy_net, path=f"model/chess_ai{episode}.pth")
            print(f"모델 저장")

        print(f"에피소드 {episode} 보상: {total_reward}")
        print("--------------------------------------------------------")


if __name__ == "__main__":
    main()