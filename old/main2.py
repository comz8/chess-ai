import pygame
import sys

# Pygame 초기화
pygame.init()

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (181, 136, 99)
W_BROWN = (237, 214, 178)
BACKGROUND = (49, 46, 43)
FONT_COLOR = (160, 160, 160)
HIGHLIGHT_COLOR = (0, 255, 0)  # 가능한 움직임 하이라이트 색상

# 기본 체스판 타일 크기 및 윈도우 크기 설정
LEFT_MARGIN = 20
BOTTOM_MARGIN = 15
WIDTH, HEIGHT = 700, 700
TILE_SIZE = WIDTH // 8
PIECE_SCALE = 1
PIECE_SCALE_CAPTURED = 0.3  # 먹힌 기물의 크기 비율
CAPTURED_AREA_WIDTH = 200  # 오른쪽에 추가된 공간의 폭
CAPTURED_TILE_SIZE = 50  # 잡힌 기물 타일 크기

# 추가된 변수
captured_pieces = {'white': [], 'black': []}  # 잡힌 기물 저장

screen = pygame.display.set_mode((WIDTH + CAPTURED_AREA_WIDTH, HEIGHT + BOTTOM_MARGIN), pygame.RESIZABLE)  # 오른쪽에 공간 추가
pygame.display.set_caption("Chess Board")

# 폰트 설정
try:
    font = pygame.font.Font('Font/Maplelight.ttf', 15)
except FileNotFoundError:
    print("Font file not found. Using default font.")
    font = pygame.font.SysFont(None, 15)

CHESS_HORSE_PIXELS = 80
PIECE_PATH = 'img/pieces.png'
pieces = {}

# 기물 이미지 불러오기 함수
def load_image():
    global pieces
    name = ['k', 'q', 'b', 'n', 'r', 'p']
    try:
        img_horse = pygame.image.load(PIECE_PATH)
        img_horse = pygame.transform.scale(img_horse, (CHESS_HORSE_PIXELS * 6, CHESS_HORSE_PIXELS * 2))
        for i in range(2):
            for j in range(6):
                cropped_region = (j * CHESS_HORSE_PIXELS, i * CHESS_HORSE_PIXELS, CHESS_HORSE_PIXELS, CHESS_HORSE_PIXELS)
                cropped = pygame.Surface((CHESS_HORSE_PIXELS, CHESS_HORSE_PIXELS), pygame.SRCALPHA)
                cropped.blit(img_horse, (0, 0), cropped_region)
                if i == 0:
                    pieces[name[j]] = cropped  # White pieces as lowercase
                else:
                    pieces[chr(ord(name[j]) - 32)] = cropped  # Black pieces as uppercase
    except pygame.error as e:
        print(f"Error loading image: {e}")
        sys.exit()

load_image()

# 이미지 크기를 타일 크기보다 작게 조정
def resize_pieces():
    for key in pieces:
        new_size = int(TILE_SIZE * PIECE_SCALE)
        pieces[key] = pygame.transform.scale(pieces[key], (new_size, new_size))

def resize_captured_pieces():
    global captured_pieces
    for color in captured_pieces:
        captured_pieces[color] = [pygame.transform.scale(pieces[piece], (int(CAPTURED_TILE_SIZE * PIECE_SCALE_CAPTURED), int(CAPTURED_TILE_SIZE * PIECE_SCALE_CAPTURED))) for piece in captured_pieces[color]]

resize_pieces()
resize_captured_pieces()

# 체스판 초기화
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

# 체스판 그리기 함수
def draw_board(screen):
    colors = [W_BROWN, BROWN]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, ((col * TILE_SIZE) + LEFT_MARGIN, (row * TILE_SIZE), TILE_SIZE, TILE_SIZE))
    for i in range(8):
        col_text = font.render(chr(ord('A') + i), True, FONT_COLOR)
        screen.blit(col_text, ((i * TILE_SIZE) + LEFT_MARGIN + TILE_SIZE // 2 - col_text.get_width() // 2, WIDTH - 3))
        row_text = font.render(str(8 - i), True, FONT_COLOR)
        screen.blit(row_text, (LEFT_MARGIN // 2 - row_text.get_width() // 2, i * TILE_SIZE + TILE_SIZE // 2 - row_text.get_height() // 2))

def draw_pieces(screen):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                new_size = int(TILE_SIZE * PIECE_SCALE)
                offset = (TILE_SIZE - new_size) // 2
                screen.blit(pieces[piece], (col * TILE_SIZE + LEFT_MARGIN + offset, row * TILE_SIZE + offset))

def draw_captured_pieces(screen):
    x_start = WIDTH + LEFT_MARGIN
    y_start_white = 20
    y_start_black = 20

    for piece in captured_pieces['white']:
        screen.blit(piece, (x_start, y_start_white))
        y_start_white += CAPTURED_TILE_SIZE

    y_start_black = y_start_white  # Start black pieces where white pieces ended
    for piece in captured_pieces['black']:
        screen.blit(piece, (x_start + CAPTURED_TILE_SIZE + 10, y_start_black))  # Slightly offset to avoid overlap
        y_start_black += CAPTURED_TILE_SIZE

def is_valid_move(start, end, piece):
    # Add your move validation logic here (like check for piece-specific moves)
    return True

def validate_rook_move(start, end):
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:  # Vertical move
        step = 1 if y2 > y1 else -1
        for y in range(y1 + step, y2, step):
            if board[y][x1] != '':
                return False
        return True
    elif y1 == y2:  # Horizontal move
        step = 1 if x2 > x1 else -1
        for x in range(x1 + step, x2, step):
            if board[y1][x] != '':
                return False
        return True
    return False

def validate_knight_move(start, end):
    x1, y1 = start
    x2, y2 = end
    return (abs(x1 - x2), abs(y1 - y2)) in [(1, 2), (2, 1)]

def validate_bishop_move(start, end):
    x1, y1 = start
    x2, y2 = end
    if abs(x1 - x2) == abs(y1 - y2):  # Diagonal move
        step_x = 1 if x2 > x1 else -1
        step_y = 1 if y2 > y1 else -1
        for i in range(1, abs(x2 - x1)):
            if board[y1 + i * step_y][x1 + i * step_x] != '':
                return False
        return True
    return False

def validate_queen_move(start, end):
    return validate_rook_move(start, end) or validate_bishop_move(start, end)

def validate_king_move(start, end):
    x1, y1 = start
    x2, y2 = end
    return max(abs(x1 - x2), abs(y1 - y2)) == 1

def handle_mouse_click(pos):
    global selected_piece, possible_moves, turn
    x, y = (pos[0] - LEFT_MARGIN) // TILE_SIZE, pos[1] // TILE_SIZE
    if 0 <= x < 8 and 0 <= y < 8:
        piece = board[y][x]
        if piece and ((piece.isupper() and turn == 'white') or (piece.islower() and turn == 'black')):  # 현재 턴의 기물만 선택 가능
            selected_piece = (x, y)
            possible_moves = get_possible_moves(x, y, piece)
        elif selected_piece:
            sx, sy = selected_piece
            if (x, y) in possible_moves:
                move_piece((sx, sy), (x, y))
                selected_piece = None
                possible_moves = []
                turn = 'white' if turn == 'black' else 'black'  # 턴 변경

def get_possible_moves(x, y, piece):
    moves = []
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]  # 8방향
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board[ny][nx] == '':
                moves.append((nx, ny))
            elif (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                moves.append((nx, ny))
                break
            else:
                break
            nx += dx
            ny += dy
    if piece.lower() == 'n':
        for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if board[ny][nx] == '' or (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                    moves.append((nx, ny))
    return moves

def move_piece(start, end):
    global captured_pieces
    sx, sy = start
    ex, ey = end
    piece = board[sy][sx]
    if board[ey][ex] != '':
        captured_piece = board[ey][ex]
        color = 'white' if captured_piece.isupper() else 'black'
        captured_pieces[color].append(captured_piece)
        resize_captured_pieces()  # Resize captured pieces
    board[ey][ex] = piece
    board[sy][sx] = ''

def main():
    global selected_piece, possible_moves, turn
    selected_piece = None
    possible_moves = []
    turn = 'black'  # 검은색 말부터 시작

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_mouse_click(event.pos)

        screen.fill(BACKGROUND)
        draw_board(screen)
        draw_pieces(screen)
        draw_captured_pieces(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
