import cv2
import numpy as np


def visualize_pose_sequence(pose_sequence):
    n_frames, n_landmarks, n_dim = pose_sequence.shape
    display_h, display_w = 300, 300
    img = np.zeros((display_h, display_w), np.uint8)
    for i in range(n_frames):
        frame = pose_sequence[i, :, :]
        for lm in range(n_landmarks):
            x = int(frame[lm, 0] * display_w)
            y = int(frame[lm, 1] * display_h)
            if x >= 0 and y >= 0:
                cv2.circle(img, (x, y), 1, (255, 0, 255), -1)

        cv2.imshow(img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
