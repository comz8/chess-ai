import chess
import random

board = chess.Board()

def simulate(board):
    sim_board = board.copy()  # 현재 보드 상태를 복사하여 시뮬레이션 보드를 만듭니다.
    
    while not sim_board.is_game_over():  # 게임이 끝날 때까지 반복합니다.
        move = random.choice(list(sim_board.legal_moves))  # 가능한 수 중에서 무작위로 선택합니다.
        sim_board.push(move)  # 선택한 수를 보드에 적용합니다.

    result = sim_board.result()  # 게임 결과를 가져옵니다.

    if result == '1-0':  # 백이 승리한 경우
        return 1
    elif result == '0-1':  # 흑이 승리한 경우
        return -1
    
    return 0  # 무승부인 경우

legal_moves = list(board.legal_moves)
print(simulate(board))