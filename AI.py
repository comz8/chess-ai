import chess
import chess.engine
import random
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, log
import os
import multiprocessing
import re
import sys



if not torch.cuda.is_available():
    print("GPU를 찾을 수 없습니다. 이 프로그램은 GPU가 필요합니다.")
    sys.exit(1)

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

torch.cuda.empty_cache()
torch.multiprocessing.set_start_method('spawn', force=True)

BEST_REWARD = float("-inf")


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 64)  
        self.value_head = nn.Linear(64, 1)    

        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
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
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            board_array.append(0)
        else:
            board_array.append(piece.piece_type if piece.color == chess.WHITE else -piece.piece_type)
    

    return torch.tensor(board_array, dtype=torch.float32, device=device)


# MCTS
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
    
    with torch.cuda.amp.autocast():  # GPU 연산 최적화
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
    return 1 if result == '1-0' else (-1 if result == '0-1' else 0)

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.wins += result
        result = -result
        node = node.parent

def train_step(policy_net, optimizer, board_state, policy, reward):
    board_tensor = fen_to_tensor(board_state)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

    with torch.cuda.amp.autocast():  # GPU 연산 최적화
        pred_policy, pred_value = policy_net(board_tensor)
        value_loss = nn.MSELoss()(pred_value, reward_tensor)
        policy_loss = -(policy * pred_policy.log()).mean()
        loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model, path="chess_ai.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="chess_ai.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))

def run_game(i):
    global BEST_REWARD

    torch.cuda.empty_cache()

    policy_net = ChessNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    try:
        load_model(policy_net, "model/chess_ai.pth")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return 0
    
    board = chess.Board()
    root = TreeNode(board)
    total_reward = 0

    while not board.is_game_over():
        selected_node = select(root)

        if not selected_node.board.is_game_over():
            selected_node = expand(selected_node, policy_net)

        reward = simulate(selected_node.board)
        backpropagate(selected_node, reward)

        board_state = selected_node.board.fen()
        board_tensor = fen_to_tensor(board_state)

        with torch.cuda.amp.autocast():  # GPU 연산 최적화
            policy, _ = policy_net(board_tensor)
        train_step(policy_net, optimizer, board_state, policy, reward)

        board = selected_node.board.copy()
        root = selected_node
        total_reward += reward

    if total_reward > BEST_REWARD and total_reward > 0:
        BEST_REWARD = total_reward
        model_path = f"model/chess_ai_{BEST_REWARD}.pth"
        save_model(policy_net, model_path)

    # 메모리 정리
    del policy_net
    del optimizer
    torch.cuda.empty_cache()

    return total_reward

def delete_lower_reward_models(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return
        
    if not os.listdir(directory):
        return

    if os.path.isfile(directory + "chess_ai.pth"):
        os.remove(directory + "chess_ai.pth")

    highest_reward = float('-inf')
    best_model_file = ""

    for filename in os.listdir(directory):
        match = re.search(r'chess_ai_([-+]?\d*\.\d+|\d+)', filename)
        if match:
            reward = float(match.group(1))
            if reward > highest_reward:
                highest_reward = reward
                best_model_file = filename

    if best_model_file:  # best_model_file이 존재할 때만 처리
        for filename in os.listdir(directory):
            if filename != best_model_file and filename.startswith("chess_ai_"):
                os.remove(os.path.join(directory, filename))

        os.rename(os.path.join(directory, best_model_file), os.path.join(directory, "chess_ai.pth"))

def main():
    global BEST_REWARD
    i = 0

    while True:
        BEST_REWARD = float('-inf')
        delete_lower_reward_models("model/")

        num_processes = min(20, torch.cuda.device_count() * 4)
        episodes = range(100)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(run_game, episodes)

        # GPU 메모리 정리
        torch.cuda.empty_cache()

        i += 1
        print(f"{i}번째 학습 완료")


if __name__ == "__main__":
    main()