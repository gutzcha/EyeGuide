import os.path as osp
import sys

import pygame

class MoveStutter:
    def __init__(self, fps, stutter=None):
        if stutter is None:
            self.move_frames = 0
            self.stop_frames = 0
        else:
            self.move_frames = round(fps * stutter['move'])
            self.stop_frames = round(fps * stutter['stop'])

        if self.stop_frames == 0:
            self.move_frames = 0

        self.fps = fps
        self.move_counter = 0
        self.stop_counter = 0
        self.phase = 'move'



    def __call__(self):
        if self.move_frames == 0:
            return True
        if self.phase == 'move':
            if self.move_counter < self.move_frames:
                self.move_counter += 1
                return True
            else:
                self.phase = 'stop'
                self.move_counter = 0
                return False
        else:
            if self.stop_counter < self.stop_frames:
                self.stop_counter += 1
                return False
            else:
                self.phase = 'move'
                self.stop_counter = 0
                return True
#
# class MovingTarget():
#     def __init__(self, img_path, start_pos, scale,fps , screen):
#         # Load the image
#         image = pygame.image.load(img_path)
#         image = pygame.transform.scale(image, scale)
#         self.x = start_pos[0]
#         self.y = start_pos[1]
#         self.fps = fps
#
#     def update_position(self):
#        pass
#
#     def draw(self):
#         self.



def move_image(path, start_pos, end_pos, scale, speed, fps=30, stutter=None):
    pygame.init()
    clock = pygame.time.Clock()
    stutter_mannager = MoveStutter(fps, stutter)


    # Load the image
    image = pygame.image.load(path)
    image = pygame.transform.scale(image, scale)

    # Create the screen
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    # Set initial position
    x, y = start_pos
    screen.blit(image, (x, y))
    pygame.display.flip()

    # Calculate movement increment based on speed and fps
    speed_x, speed_y = speed
    increment_x = speed_x / fps
    increment_y = speed_y / fps

    stutter_counter = 0

    while True:
        for event in pygame.event.get():
            if (
                    event.type == pygame.QUIT or
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
            ):
                pygame.quit()
                sys.exit()

        # Update position
        update_move = stutter_mannager()
        if update_move:
            x += increment_x
            y += increment_y

            # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the image at the updated position
        screen.blit(image, (x, y))
        pygame.display.flip()

        # debug
        # print(f'x_pos:{x} y_pos:{y}')
        # print(f'img_width:{-image.get_width()} img_height:{-image.get_height()}')
        # print(f'screen_width:{screen.get_width()} screen_height:{screen.get_height()}')

        # Check termination conditions
        if (
                x < -image.get_width() or
                y < -image.get_height() or
                x > screen.get_width() or
                y > screen.get_height() or
                x > end_pos[0] or
                y > end_pos[1]
        ):
            break

        clock.tick(fps)

    pygame.quit()
    sys.exit()


img_path = osp.join('assets', 'target.png')
start_x = 0
start_y = 500

end_x = 1500
end_y = 500

width = 50
height = 50

speed_x = 200
speed_y = 0

fps = 30
stutter = {'move': 2, 'stop': 0.1}  # sec

move_image(img_path, (start_x, start_y), (end_x, end_y), (width, height), (speed_x, speed_y), fps, stutter=stutter)
