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
CHECK_COLOR = (255, 0, 0)  # 체크 시 색상
MOVE_DOT_COLOR = (0, 255, 0)  # 이동 가능한 위치 색상

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
selected_piece = None  # 선택된 기물 위치
possible_moves = []  # 이동 가능한 위치 목록
turn = 'white'  # 턴을 추적
king_positions = {'white': (4, 0), 'black': (4, 7)}  # 각 색의 왕 위치 저장

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

def draw_highlight(screen, selected_piece, possible_moves, check_color=None):
    if selected_piece:
        x, y = selected_piece
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, (x * TILE_SIZE + LEFT_MARGIN, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)
    for move in possible_moves:
        mx, my = move
        pygame.draw.circle(screen, MOVE_DOT_COLOR, (mx * TILE_SIZE + LEFT_MARGIN + TILE_SIZE // 2, my * TILE_SIZE + TILE_SIZE // 2), 5)
    if check_color:  # Highlight the king's position if in check
        king_x, king_y = king_positions[turn]
        pygame.draw.rect(screen, check_color, (king_x * TILE_SIZE + LEFT_MARGIN, king_y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

def get_possible_moves(x, y, piece):
    moves = []
    
    if piece.lower() == 'r':  # Rook
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if board[ny][nx] == '':
                        moves.append((nx, ny))
                    elif (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                        moves.append((nx, ny))
                        break
                    else:
                        break
                else:
                    break

    elif piece.lower() == 'n':  # Knight
        for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if board[ny][nx] == '' or (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                    moves.append((nx, ny))

    elif piece.lower() == 'b':  # Bishop
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if board[ny][nx] == '':
                        moves.append((nx, ny))
                    elif (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                        moves.append((nx, ny))
                        break
                    else:
                        break
                else:
                    break

    elif piece.lower() == 'q':  # Queen
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if board[ny][nx] == '':
                        moves.append((nx, ny))
                    elif (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                        moves.append((nx, ny))
                        break
                    else:
                        break
                else:
                    break

    elif piece.lower() == 'k':  # King
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if board[ny][nx] == '' or (board[ny][nx].islower() and piece.isupper()) or (board[ny][nx].isupper() and piece.islower()):
                    moves.append((nx, ny))
    
    elif piece.lower() == 'p':  # Pawn
        direction = -1 if piece.isupper() else 1
        start_row = 6 if piece.isupper() else 1
        if 0 <= y + direction < 8:
            if board[y + direction][x] == '':
                moves.append((x, y + direction))
                if y == start_row and board[y + 2 * direction][x] == '':
                    moves.append((x, y + 2 * direction))
        if 0 <= x + 1 < 8 and 0 <= y + direction < 8 and board[y + direction][x + 1] != '' and board[y + direction][x + 1].islower() != piece.islower():
            moves.append((x + 1, y + direction))
        if 0 <= x - 1 < 8 and 0 <= y + direction < 8 and board[y + direction][x - 1] != '' and board[y + direction][x - 1].islower() != piece.islower():
            moves.append((x - 1, y + direction))

    return moves

def is_in_check(color):
    king_x, king_y = king_positions[color]
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '' and ((piece.islower() and color == 'white') or (piece.isupper() and color == 'black')):
                moves = get_possible_moves(col, row, piece)
                if (king_x, king_y) in moves:
                    return True
    return False

# 게임 루프
while True:
    screen.fill(BACKGROUND)
    draw_board(screen)
    draw_pieces(screen)

    # 체크 여부 확인
    if is_in_check(turn):
        draw_highlight(screen, None, [], CHECK_COLOR)
    else:
        draw_highlight(screen, selected_piece, possible_moves)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 왼쪽 클릭
            mouse_x, mouse_y = event.pos
            # 클릭한 타일 계산
            col = (mouse_x - LEFT_MARGIN) // TILE_SIZE
            row = mouse_y // TILE_SIZE

            if 0 <= col < 8 and 0 <= row < 8:
                piece = board[row][col]
                if selected_piece is None:  # 기물 선택
                    if (piece.islower() and turn == 'white') or (piece.isupper() and turn == 'black'):
                        selected_piece = (col, row)
                        possible_moves = get_possible_moves(col, row, piece)
                else:  # 기물 이동
                    if (col, row) in possible_moves:
                        captured_piece = board[row][col]
                        if captured_piece:  # 기물 잡기
                            captured_pieces['black' if captured_piece.islower() else 'white'].append(captured_piece)
                        board[row][col] = piece
                        board[selected_piece[1]][selected_piece[0]] = ''  # 이전 위치 비우기
                        # 왕의 위치 업데이트
                        if piece.lower() == 'k':
                            king_positions[turn] = (col, row)
                        selected_piece = None
                        possible_moves = []
                        # 턴 변경
                        turn = 'black' if turn == 'white' else 'white'
                    else:
                        selected_piece = None
                        possible_moves = []

    # 잡힌 기물 그리기
    for idx, piece in enumerate(captured_pieces['white']):
        screen.blit(piece, (WIDTH + 20, idx * CAPTURED_TILE_SIZE))
    for idx, piece in enumerate(captured_pieces['black']):
        screen.blit(piece, (WIDTH + 20, (idx + 1) * CAPTURED_TILE_SIZE))

    pygame.display.flip()
