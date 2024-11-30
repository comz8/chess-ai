import chess
import chess.engine
import engine
import torch.optim as optim
import os, sys

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame



def init_gui():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((480, 480))
    pygame.display.set_caption("Chess Game")
    return screen


def draw_board(screen, board, valid_moves=None):
    colors = [pygame.Color("white"), pygame.Color("gray")]

    # Draw squares
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * 60, row * 60, 60, 60))

    path = "assets\\"


    piece_images = {
        f[:2]: pygame.image.load(path + f)
        for f in os.listdir(path) if f.endswith('.png')
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'w' if piece.color == chess.WHITE else 'b'
            piece_name = f"{color}{piece.symbol().lower()}"
            screen.blit(piece_images[piece_name], (chess.square_file(square) * 60, (7 - chess.square_rank(square)) * 60))

    if valid_moves:
        for move in valid_moves:
            x, y = chess.square_file(move), 7 - chess.square_rank(move)
            pygame.draw.circle(screen, pygame.Color("green"), (x * 60 + 30, y * 60 + 30), 5)

    pygame.display.flip()


def mouse_to_square(mouse_pos):
    x, y = mouse_pos
    col = x // 60
    row = 7 - (y // 60)
    return chess.square(col, row)

# Handle human move
def handle_human_move(board, screen):
    selected_square = None
    valid_moves = []
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                square = mouse_to_square(mouse_pos)

                if selected_square is None:
                    piece = board.piece_at(square)
                    if piece and piece.color == chess.WHITE:
                        selected_square = square
                        valid_moves = [move.to_square for move in board.legal_moves if move.from_square == selected_square]
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        return
                    else:
                        selected_square = None

            # Cancel selection
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                selected_square = None  

        draw_board(screen, board, valid_moves)


# Game loop
def game():
    policy_net = engine.ChessNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    engine.load_model(policy_net, "chess_ai.pth")
    
    board = chess.Board()
    root = engine.TreeNode(board)
    
    screen = init_gui()

    while not board.is_game_over():
        draw_board(screen, board)

        # Human move (only white plays for now)
        if board.turn == chess.WHITE:
            handle_human_move(board, screen)

        if not board.is_game_over():
            # AI move
            root = engine.TreeNode(board)
            selected_node = engine.select(root)
            if not selected_node.board.is_game_over():
                selected_node = engine.expand(selected_node, policy_net)

            reward = engine.simulate(selected_node.board)
            engine.backpropagate(selected_node, reward)

            board_state = selected_node.board.fen()
            board_tensor = engine.fen_to_tensor(board_state)
            policy, _ = policy_net(board_tensor)
            engine.train_step(policy_net, optimizer, board_state, policy, reward)

            board = selected_node.board.copy()
            root = selected_node


    print("Game over:", board.result())
    pygame.quit()

if __name__ == "__main__":
    game()
