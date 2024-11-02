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

# 기본 체스판 타일 크기 및 윈도우 크기 설정
LEFT_MARGIN = 20
BOTTOM_MARGIN = 15

WIDTH, HEIGHT = 700, 700
TILE_SIZE = WIDTH // 8

# 말 크기 비율 설정
PIECE_SCALE = 1

screen = pygame.display.set_mode((WIDTH + LEFT_MARGIN, HEIGHT + BOTTOM_MARGIN), pygame.RESIZABLE)
pygame.display.set_caption("Chess Board")

# 폰트 설정 (Ensure the font path is correct)
try:
    font = pygame.font.Font('Font/Maplelight.ttf', 15)
except FileNotFoundError:
    print("Font file not found. Using default font.")
    font = pygame.font.SysFont(None, 15)

CHESS_HORSE_PIXELS = 80
PIECE_PATH = 'img/pieces.png'

pieces = {}

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
print(pieces)

# 이미지 크기를 타일 크기보다 작게 조정
def resize_pieces():
    for key in pieces:
        new_size = int(TILE_SIZE * PIECE_SCALE)
        pieces[key] = pygame.transform.scale(pieces[key], (new_size, new_size))

resize_pieces()

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

# 현재 드래그 중인 말 정보
dragging_piece = None
dragging_pos = (0, 0)
start_pos = (0, 0)

# 메인 루프
def main():
    global TILE_SIZE, screen, dragging_piece, dragging_pos, start_pos

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = (x - LEFT_MARGIN) // TILE_SIZE
                row = y // TILE_SIZE

                if 0 <= col < 8 and 0 <= row < 8:
                    dragging_piece = board[row][col]
                    if dragging_piece:
                        dragging_pos = (x - LEFT_MARGIN, y)
                        start_pos = (row, col)
                        board[row][col] = ''  # 원래 위치를 비웁니다.

            elif event.type == pygame.MOUSEBUTTONUP:
                if dragging_piece:
                    x, y = event.pos
                    col = (x - LEFT_MARGIN) // TILE_SIZE
                    row = y // TILE_SIZE

                    if 0 <= col < 8 and 0 <= row < 8:
                        board[row][col] = dragging_piece
                    else:
                        # 범위를 벗어나면 원래 위치로 되돌립니다.
                        board[start_pos[0]][start_pos[1]] = dragging_piece

                    dragging_piece = None

            elif event.type == pygame.MOUSEMOTION and dragging_piece:
                dragging_pos = event.pos

            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                TILE_SIZE = min(WIDTH, HEIGHT) // 10  # Adjust to keep tiles square
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                
                # 윈도우 크기에 맞게 말 크기 조정
                resize_pieces()

        screen.fill(BACKGROUND)
        draw_board(screen)
        draw_pieces(screen)

        if dragging_piece:
            new_size = int(TILE_SIZE * PIECE_SCALE)
            screen.blit(pieces[dragging_piece], (dragging_pos[0] - new_size // 2, dragging_pos[1] - new_size // 2))

        # 화면 업데이트
        pygame.display.flip()

if __name__ == "__main__":
    main()
