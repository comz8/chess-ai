import chess
import chess.engine
import engine
import torch.optim as optim

import time

import os, sys


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame



# # 사용자 입력 처리 함수
# def user_move(board):
#     while True:
#         try:
#             move = input("당신의 수를 입력하세요 (예: e2e4): ")
#             move = chess.Move.from_uci(move)
#             if move in board.legal_moves:
#                 board.push(move)
#                 return
#             else:
#                 print("유효하지 않은 수입니다. 다시 시도하세요.")
#         except Exception as e:
#             print("입력 오류:", e)

# GUI 초기화
def init_gui():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("체스 게임")

    return screen

# 체스판 그리기
def draw_board(screen, board):
    colors = [pygame.Color("white"), pygame.Color("gray")]

    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * 60, row * 60, 60, 60))

    path = "AI\\assets\\"

    # 기물 이미지 로드
    piece_images = {
        f[:2]: pygame.image.load(path + f)
        for f in os.listdir(path) if f.endswith('.png')
    }


    # 기물 그리기
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'w' if piece.color == chess.WHITE else 'b'
            piece_name = f"{color}{piece.symbol().lower()}"
            screen.blit(piece_images[piece_name], (chess.square_file(square) * 60, (7 - chess.square_rank(square)) * 60))

    pygame.display.flip()


def game():
    policy_net = engine.ChessNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    engine.load_model(policy_net, "chess_ai.pth")
    
    board = chess.Board()
    root = engine.TreeNode(board)
    
    screen = init_gui()

    while not board.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        draw_board(screen, board)

        root = engine.TreeNode(board)

        if not board.is_game_over():
            selected_node = engine.select(root)
            if not selected_node.board.is_game_over():
                selected_node = engine.expand(selected_node, policy_net)

            # 시뮬레이션을 통해 보드의 결과를 가져옵니다.
            reward = engine.simulate(selected_node.board)
            engine.backpropagate(selected_node, reward)

            # 모델 학습
            board_state = selected_node.board.fen()
            board_tensor = engine.fen_to_tensor(board_state)

            # 정책 네트워크에 전달
            policy, _ = policy_net(board_tensor)
            engine.train_step(policy_net, optimizer, board_state, policy, reward)

            # board를 업데이트
            board = selected_node.board.copy()
            root = selected_node

            time.sleep(1)

    print("게임 종료:", board.result())
    pygame.quit()


if __name__ == "__main__":
    game()
