import pygame
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# 초기화
pygame.init()

# 상수 정의
LEFT_MARGIN = 20
BOTTOM_MARGIN = 20

WIDTH, HEIGHT = 720, 720
SQUARE_SIZE = (WIDTH - LEFT_MARGIN) // 8
BROWN = (181, 136, 99)
W_BROWN = (237, 214, 178)
BACKGROUND = (49, 46, 43)
FONT_COLOR = (145, 144, 142)


font = pygame.font.SysFont("arial", 15)

# 화면 설정
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Board")

def draw_board():
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                pygame.draw.rect(screen, W_BROWN, (LEFT_MARGIN + col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            else:
                pygame.draw.rect(screen, BROWN, (LEFT_MARGIN + col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

        row_text = font.render(row + 1, True, FONT_COLOR)

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BACKGROUND)
        draw_board()
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()