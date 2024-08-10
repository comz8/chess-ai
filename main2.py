import pygame
import sys

# Pygame 초기화
pygame.init()

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# 기본 체스판 타일 크기 및 윈도우 크기 설정
TILE_SIZE = 80
BOARD_SIZE = TILE_SIZE * 8
MARGIN = 50  # 행과 열 이름을 표시할 여유 공간

# 말 크기 비율 설정
PIECE_SCALE = 0.7  # 타일 크기의 70%

# 윈도우 설정 (크기 조절 가능)
screen = pygame.display.set_mode((BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN), pygame.RESIZABLE)
pygame.display.set_caption("Chess Board")

# 폰트 설정
font = pygame.font.SysFont('Arial', 24)

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

# 체스 말 배치 함수
def place_piece(screen, piece, x, y):
    new_size = int(TILE_SIZE * PIECE_SCALE)
    offset = (TILE_SIZE - new_size) // 2
    screen.blit(pieces[piece], (x * TILE_SIZE + MARGIN + offset, y * TILE_SIZE + offset))

# 체스판 그리기 함수
def draw_board(screen):
    colors = [WHITE, GRAY]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * TILE_SIZE + MARGIN, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    # 행과 열 이름 그리기
    for i in range(8):
        # 열 이름 (A-H) - 체스판 위쪽에 표시
        col_text = font.render(chr(ord('A') + i), True, BLACK)
        screen.blit(col_text, (i * TILE_SIZE + MARGIN + TILE_SIZE // 2 - col_text.get_width() // 2, BOARD_SIZE))

        # 행 이름 (1-8) - 체스판 왼쪽에 표시
        row_text = font.render(str(8 - i), True, BLACK)
        screen.blit(row_text, (MARGIN // 2 - row_text.get_width() // 2, i * TILE_SIZE + TILE_SIZE // 2 - row_text.get_height() // 2))

# 메인 루프
def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                global TILE_SIZE, BOARD_SIZE, screen
                TILE_SIZE = event.w // 10
                BOARD_SIZE = TILE_SIZE * 8
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
                # 윈도우 크기에 맞게 말 크기 조정
                resize_pieces()

        # 배경 지우기
        screen.fill(WHITE)

        # 체스판 그리기
        draw_board(screen)

        # 체스 말 배치 (예시)
        place_piece(screen, 'K', 4, 4)  # 킹을 E5에 배치
        place_piece(screen, 'Q', 3, 4)  # 퀸을 D5에 배치
        place_piece(screen, 'R', 0, 0)  # 룩을 A1에 배치
        place_piece(screen, 'B', 2, 0)  # 비숍을 C1에 배치
        place_piece(screen, 'N', 1, 2)  # 나이트를 B3에 배치
        place_piece(screen, 'P', 4, 6)  # 폰을 E7에 배치

        # 화면 업데이트
        pygame.display.flip()

if __name__ == "__main__":
    main()
