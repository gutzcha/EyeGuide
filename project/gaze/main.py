import pygame
import sys
from project.utils.constants import COLORS


# Create the screen
WIN = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

FPS = 30
bg_color = COLORS['WHITE']

def draw_window():
    WIN.fill(bg_color)
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    pygame.display.set_caption("Calibrate Eye Tracker")

    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if (
                    event.type == pygame.QUIT or
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
            ):
                run = False
        draw_window()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
