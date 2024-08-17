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

# 체스 말 배치 (2D 배열로 표현)
board = [
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],  # 대문자 'P' 사용
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],  # 대문자 'P'로 변경
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
]

# 현재 드래그 중인 말 정보
dragging_piece = None
dragging_pos = (0, 0)
start_pos = (0, 0)

# 체스 말 이동 규칙을 확인하는 함수
def is_valid_move(piece, start, end):
    start_row, start_col = start
    end_row, end_col = end

    if piece.lower() == 'p':  # 폰
        direction = 1 if piece.islower() else -1  # 흑색 폰은 아래로, 백색 폰은 위로
        if end_col == start_col:  # 수직 이동
            if end_row == start_row + direction and board[end_row][end_col] == '':
                return True  # 1칸 이동
            if (start_row == 1 and piece.islower() and end_row == start_row + 2) and board[start_row + 1][end_col] == '':
                return True  # 2칸 이동
            if (start_row == 6 and piece.isupper() and end_row == start_row - 2) and board[start_row - 1][end_col] == '':
                return True
        elif abs(end_col - start_col) == 1 and end_row == start_row + direction:
            return True  # 대각선 먹기

    elif piece.lower() == 'r':  # 룩
        if start_row == end_row or start_col == end_col:
            return not is_path_blocked(start, end)

    elif piece.lower() == 'n':  # 나이트
        if (abs(start_row - end_row) == 2 and abs(start_col - end_col) == 1) or (abs(start_row - end_row) == 1 and abs(start_col - end_col) == 2):
            return True

    elif piece.lower() == 'b':  # 비숍
        if abs(start_row - end_row) == abs(start_col - end_col):
            return not is_path_blocked(start, end)

    elif piece.lower() == 'q':  # 퀸
        if (start_row == end_row or start_col == end_col) or (abs(start_row - end_row) == abs(start_col - end_col)):
            return not is_path_blocked(start, end)

    elif piece.lower() == 'k':  # 킹
        if abs(start_row - end_row) <= 1 and abs(start_col - end_col) <= 1:
            return True

    return False

# 이동 경로가 막혀 있는지 확인하는 함수
def is_path_blocked(start, end):
    start_row, start_col = start
    end_row, end_col = end

    if start_row == end_row:  # 수평 이동
        step = 1 if end_col > start_col else -1
        for col in range(start_col + step, end_col, step):
            if board[start_row][col] != '':
                return True
    elif start_col == end_col:  # 수직 이동
        step = 1 if end_row > start_row else -1
        for row in range(start_row + step, end_row, step):
            if board[row][start_col] != '':
                return True
    return False

# 체스판 그리기 함수
def draw_board(screen):
    colors = [WHITE, GRAY]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * TILE_SIZE + MARGIN, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    # 행과 열 이름 그리기
    for i in range(8):
        # 열 이름 (A-H) - 체스판 아래쪽에 표시
        col_text = font.render(chr(ord('A') + i), True, BLACK)
        screen.blit(col_text, (i * TILE_SIZE + MARGIN + TILE_SIZE // 2 - col_text.get_width() // 2, BOARD_SIZE))

        # 행 이름 (1-8) - 체스판 왼쪽에 표시
        row_text = font.render(str(8 - i), True, BLACK)
        screen.blit(row_text, (MARGIN // 2 - row_text.get_width() // 2, i * TILE_SIZE + TILE_SIZE // 2 - row_text.get_height() // 2))

# 체스 말 그리기 함수
def draw_pieces(screen):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                new_size = int(TILE_SIZE * PIECE_SCALE)
                offset = (TILE_SIZE - new_size) // 2
                screen.blit(pieces[piece], (col * TILE_SIZE + MARGIN + offset, row * TILE_SIZE + offset))

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
                col = (x - MARGIN) // TILE_SIZE
                row = y // TILE_SIZE

                if 0 <= col < 8 and 0 <= row < 8:
                    dragging_piece = board[row][col]
                    if dragging_piece:
                        dragging_pos = (x - MARGIN, y)
                        start_pos = (row, col)
                        board[row][col] = ''  # 원래 위치를 비웁니다.

            elif event.type == pygame.MOUSEBUTTONUP:
                if dragging_piece:
                    x, y = event.pos
                    col = (x - MARGIN) // TILE_SIZE
                    row = y // TILE_SIZE

                    if 0 <= col < 8 and 0 <= row < 8:
                        if is_valid_move(dragging_piece, start_pos, (row, col)):
                            board[row][col] = dragging_piece  # 유효한 이동이면 이동
                        else:
                            board[start_pos[0]][start_pos[1]] = dragging_piece  # 유효하지 않으면 원래 위치로 되돌림

                    dragging_piece = None

            elif event.type == pygame.MOUSEMOTION and dragging_piece:
                dragging_pos = event.pos

            elif event.type == pygame.VIDEORESIZE:
                TILE_SIZE = event.w // 10  # global 선언 이전에 변수를 사용하지 않도록 수정
                BOARD_SIZE = TILE_SIZE * 8
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
                # 윈도우 크기에 맞게 말 크기 조정
# 배경 지우기
screen.fill(WHITE)

# 체스판 그리기
draw_board(screen)

# 체스 말 그리기
draw_pieces(screen)

# 드래그 중인 말 그리기
if dragging_piece:
            new_size = int(TILE_SIZE * PIECE_SCALE)
            screen.blit(pieces[dragging_piece], (dragging_pos[0] - new_size // 2, dragging_pos[1] - new_size // 2))

# 화면 업데이트
pygame.display.flip()

if __name__ == "__main__":
    main()