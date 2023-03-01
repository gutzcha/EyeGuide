'''
Rule based classification

This is part of the implementation of the gesture classifier without ML, based on rules
Here a list of gestures and the rules for classifying:

'''
from utils.constants import important_keypoints, left_eye_keypoints, right_eye_keypoints
from math import dist

class BaseRuleGesture():
    def __init__(self, conditions):
        self.conditions = conditions

        pass

    def __call__(self, lm):
        pass


class Blinking(BaseRuleGesture):
    def __init__(self, conditions, side):
        super().__init__(conditions=conditions)
        self.side = side


    def set_lm(self):
        if self.side == 'left':
            keypoints = left_eye_keypoints
        else:
            keypoints = right_eye_keypoints




    def __call__(self, lm):



        lm = lm[self.lm_inds,:,:]

