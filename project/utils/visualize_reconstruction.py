import cv2
import numpy as np
from utils.constants import LEFT_EYE, RIGHT_EYE, right_eye_keypoints, left_eye_keypoints, COLORS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
from utils.constants import TRAINED_LANDMARKS

def visualize_pose_sequence(pose_sequence, inflate_ratio=1.0, center=True, loops=0, landmark_map=None):
    pose_sequence = pose_sequence.numpy()
    n_files, n_frames, n_landmarks, n_dim = pose_sequence.shape
    n_frames *= n_files
    pose_sequence = pose_sequence.reshape(1, n_frames, n_landmarks, n_dim).squeeze()

    display_h, display_w = 800, 1200

    # for i in range(n_frames):
    loop = 0
    i = 0

    max_x = max(pose_sequence[:, :, 0].flatten())
    min_x = min(pose_sequence[:, :, 0].flatten())
    mean_x = np.mean(pose_sequence[:, :, 0].flatten())
    max_y = max(pose_sequence[:, :, 1].flatten())
    min_y = min(pose_sequence[:, :, 1].flatten())
    mean_y = np.mean(pose_sequence[:, :, 1].flatten())

    if landmark_map is None:
        landmark_map = [None] * n_landmarks

    while True:
        img = np.full((display_h, display_w, 3), 0).astype(np.uint8)
        frame = pose_sequence[i, :, :]
        for lm, lmap in zip(range(n_landmarks), landmark_map):
            color = COLORS['GREEN']
            if lmap is not None:
                if lmap in LEFT_EYE:
                    color = COLORS['RED']
                elif lmap in RIGHT_EYE:
                    color = COLORS['BLUE']
                elif (lmap in right_eye_keypoints) or (lmap in left_eye_keypoints):
                    color = COLORS['YELLOW']
            x = frame[lm, 0]
            y = frame[lm, 1]

            if inflate_ratio == 0:
                x = (x - min_x) / (max_x - min_x) * display_w
                y = (y - min_y) / (max_y - min_y) * display_h
            else:
                x *= inflate_ratio
                y *= inflate_ratio

                if center:
                    x = x + display_w / 2 - mean_x * inflate_ratio
                    y = y + display_h / 2 - mean_y * inflate_ratio

            x = int(x)
            y = int(y)
            if x >= 0 and y >= 0:
                cv2.circle(img, (x, y), 1, color, -1)

        i += 1
        cv2.imshow('reconstruction', img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        if i >= n_frames - 1:
            if loop <= loops:
                loop += 1
                i = 0
            else:
                cv2.destroyAllWindows()
                break


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, pose_sequence, inflate_ratio, center):

        self.inflate_ratio = inflate_ratio
        self.center = center
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        pose_sequence = pose_sequence.numpy()
        b, f, lm, d = pose_sequence.shape
        pose_sequence = pose_sequence.reshape(-1, b*f, lm, d ).squeeze()
        self.pose_sequence = pose_sequence
        print(pose_sequence.shape)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=200,
                                           init_func=self.setup_plot, frames=b*f, blit=True, repeat=False)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, c = self.data_stream(0)
        self.scat = self.ax.scatter(x, y)
        self.ax.axis([-10, 1000, -10, 1000])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self, i):
        pose_sequence = self.pose_sequence
        inflate_ratio = self.inflate_ratio
        center = self.center

        # pose_sequence = pose_sequence.numpy()
        n_frames, n_landmarks, n_dim = pose_sequence.shape
        # n_frames *= n_files
        # pose_sequence = pose_sequence.reshape(1, n_frames, n_landmarks, n_dim).squeeze()

        display_h, display_w = 800, 1200

        # for i in range(n_frames):
        loop = 0

        max_x = max(pose_sequence[:, :, 0].flatten())
        min_x = min(pose_sequence[:, :, 0].flatten())
        mean_x = np.mean(pose_sequence[:, :, 0].flatten())
        max_y = max(pose_sequence[:, :, 1].flatten())
        min_y = min(pose_sequence[:, :, 1].flatten())
        mean_y = np.mean(pose_sequence[:, :, 1].flatten())

        single_frame_x = []
        single_frame_y = []
        single_frame_c = []

        img = np.full((display_h, display_w, 3), 0).astype(np.uint8)
        frame = pose_sequence[i, :, :]


        for lm in range(n_landmarks):
            if lm in LEFT_EYE:
                color = COLORS['RED']
            elif lm in RIGHT_EYE:
                color = COLORS['BLUE']
            elif (lm in right_eye_keypoints) or (lm in left_eye_keypoints):
                color = COLORS['YELLOW']
            else:
                color = COLORS['GREEN']
            x = frame[lm, 0]
            y = frame[lm, 1]

            if inflate_ratio == 0:
                x = (x - min_x) / (max_x - min_x) * display_w
                y = (y - min_y) / (max_y - min_y) * display_h
            else:
                x *= inflate_ratio
                y *= inflate_ratio

                if center:
                    x = x + display_w / 2 - mean_x * inflate_ratio
                    y = y + display_h / 2 - mean_y * inflate_ratio

            x = int(x)
            y = int(y)
            single_frame_x.append(x)
            single_frame_y.append(y)
            single_frame_c = color
        return single_frame_x, single_frame_y, single_frame_c

    def update(self, i):
        """Update the scatter plot."""
        x, y, col = self.data_stream(i)
        x = [1 for k in x]
        y =  [1 for k in y]

        # Set x and y data...
        data = np.hstack((x, y))
        self.scat.set_offsets(data)
        # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

if __name__ == '__main__':
    from data.dataset import get_dataloader
    from utils.constants import TRAINED_LANDMARKS

    n_landmarks = len(TRAINED_LANDMARKS)
    train_ds, train_dl = get_dataloader(data_path=None, n_landmarks=n_landmarks, sample_size_limit=64, batch_size=4,
                                        n_frames=128)

    dl_iter = iter(train_dl)
    pose_sequence = next(dl_iter)
    # pose_sequence = next(dl_iter)
    # pose_sequence = next(dl_iter)
    # pose_sequence = next(dl_iter)

    # a = AnimatedScatter(pose_sequence=pose_sequence, inflate_ratio=1, center=True)
    visualize_pose_sequence(pose_sequence, inflate_ratio=1.5, center=True, loops=0)
    plt.show()