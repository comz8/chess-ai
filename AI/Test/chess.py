from collections import namedboardle
from random import choice
from monte_carlo import MCTS, Node

_CBDT = namedboardle("ChessBoard", "board turn winner terminal") # Chess BoarD boardle


# Inheriting from a namedboardle is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class ChessBoard(_CBDT, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.board) if value is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        
        empty_spots = [i for i, value in enumerate(board.board) if value is None]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        
        if board.winner is board.turn:
            raise RuntimeError(f"reward called on unreachable board {board}")
        
        if board.turn is (not board.winner):
            return 0 # 상대가 이김
        
        if board.winner is None:
            return 0.5  # Board is a tie

        # 어느쪽도 아님
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        board = board.board[:index] + (board.turn,) + board.board[index + 1 :]
        turn = not board.turn
        winner = _find_winner(board)
        is_terminal = (winner is not None) or not any(v is None for v in board)

        return ChessBoard(board, turn, winner, is_terminal)

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.board[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def new_chess_board():
    board = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
    ]

    return ChessBoard(board = board, turn = True, winner = None, terminal = False)


def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(board):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = board[i1], board[i2], board[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    
    return None



def play_game():
    tree = MCTS()
    board = new_chess_board()

    print(board.to_pretty_string())

    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)
        
        if board.board[index] is not None:
            raise RuntimeError("Invalid move")
        
        board = board.make_move(index)
        print(board.to_pretty_string())
        
        if board.terminal:
            break

        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.do_rollout(board)
        
        board = tree.choose(board)
        
        print(board.to_pretty_string())
        
        if board.terminal:
            break





if __name__ == "__main__":
    play_game()