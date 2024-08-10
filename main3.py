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
PIECE_SCALE = 0.7  # 타일 크기의 70%

# 윈도우 설정 (크기 조절 가능)
screen = pygame.display.set_mode((WIDTH + LEFT_MARGIN, HEIGHT + BOTTOM_MARGIN))
pygame.display.set_caption("Chess Board")

# 폰트 설정
font = pygame.font.Font('Font/Maplelight.ttf', 15)

# 체스 말 이미지 로드
pieces = {
    'K': pygame.image.load('blackking.png'),
    'Q': pygame.image.load('blackqueen.png'),
    'R': pygame.image.load('blackrook.png'),
    'B': pygame.image.load('blackbishop.png'),
    'N': pygame.image.load('blackknight.png'),
    'P': pygame.image.load('blackpawn.png'),
}


# 이미지 크기를 타일 크기보다 작게 조정
def resize_pieces():
    for key in pieces:
        new_size = int(TILE_SIZE * PIECE_SCALE)
        pieces[key] = pygame.transform.scale(pieces[key], (new_size, new_size))


resize_pieces()

# 체스 말 배치 (2D 배열로 표현)
board = [
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
]

# 현재 드래그 중인 말 정보
dragging_piece = None
dragging_pos = (0, 0)
start_pos = (0, 0)

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

# 메인 루프
def main():
    global TILE_SIZE, BOARD_SIZE, screen, dragging_piece, dragging_pos, start_pos  # global 선언을 함수의 시작부분에 배치

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
                TILE_SIZE = event.w // 10  # global 선언 이전에 변수를 사용하지 않도록 수정
                BOARD_SIZE = TILE_SIZE * 8
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
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