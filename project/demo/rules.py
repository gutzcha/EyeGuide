'''
Rule based classification

This is part of the implementation of the gesture classifier without ML, based on rules
Here a list of gestures and the rules for classifying:

'''
from utils.constants import important_keypoints, left_eye_keypoints, right_eye_keypoints, LEFT_EYE, RIGHT_EYE
import math


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


class BaseRuleGesture():
    def __init__(self, th):
        self.th = th
        pass


import numpy as np


class Blinking(BaseRuleGesture):
    def __init__(self, th, right_indices, left_indices):
        super().__init__(th)
        self.right_indices = right_indices
        self.left_indices = left_indices

    def __call__(self, landmarks):
        landmarks = np.array([L['results'] for L in landmarks])

        # Right eyes
        # horizontal line
        rh_right = landmarks[self.right_indices[0]]
        rh_left = landmarks[self.right_indices[8]]
        # vertical line
        rv_top = landmarks[self.right_indices[12]]
        rv_bottom = landmarks[self.right_indices[4]]
        # draw lines on right eyes
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[self.left_indices[0]]
        lh_left = landmarks[self.left_indices[8]]

        # vertical line
        lv_top = landmarks[self.left_indices[12]]
        lv_bottom = landmarks[self.left_indices[4]]

        rh_distance = euclaideanDistance(rh_right, rh_left)
        rv_distance = euclaideanDistance(rv_top, rv_bottom)

        lv_distance = euclaideanDistance(lv_top, lv_bottom)
        lh_distance = euclaideanDistance(lh_right, lh_left)

        re_ratio = rh_distance / rv_distance
        le_ratio = lh_distance / lv_distance

        return (re_ratio > self.th), (le_ratio > self.th)


class TrippelWink(BaseRuleGesture):
    def __init__(self, th=5.5, right_indices=RIGHT_EYE, left_indices=LEFT_EYE):
        super().__init__(th)
        self.blink_detector = Blinking(self.th, right_indices, left_indices)
        self._r_counter = 0  # count number of frames while winking
        self._l_counter = 0  # count number of frames while winking
        self._blink_counter = 0
        self.wait = 2

        self.wink_r = False
        self.wink_l = False

    def reset_state(self):
        self._r_counter = 0
        self._l_counter = 0
        self._blink_counter = 0
        self.wink_l = False
        self.wink_r = False

    def __call__(self, landmarks):

        """
        Determine if there was a triple winking right->left->right
        :param landmarks:
        :return boolian:
        """

        r_wink, l_wink = self.blink_detector(landmarks)
        if r_wink and l_wink:  # this is a blink not a wink
            self._blink_counter += 1
            if self._r_counter > self.wait:
                self.reset_state()
            # print('State reset blink')
            # print(f'res right:{r_wink}, res left:{l_wink}')
            return False

        if (not r_wink) and (not l_wink):
            return False

        if r_wink:  # only right eye is winking
            self._r_counter += 1
            if self._r_counter > self.wait:  # if the eye was shut long enough, this is a wink
                self.wink_r = True

            if self.wink_l:  # this is the second right wink
                self.reset_state()
                # print('State reset success')
                return True
            else:
                return False

        if l_wink:
            if not self.wink_r:  # the first eye to wink must be the right
                self.reset_state()
                # print('State reset first left blink')
            else:
                self._l_counter += 1  # if the right eye already winked, advance counter
                if self._l_counter > self.wait:
                    self.wink_l = True
            return False
