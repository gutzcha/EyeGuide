'''
Rule based classification

This is part of the implementation of the gesture classifier without ML, based on rules
Here a list of gestures and the rules for classifying:

'''
from typing import List
from collections import defaultdict
from utils.constants import important_keypoints, left_eye_keypoints, right_eye_keypoints, LEFT_EYE, RIGHT_EYE
import math
import numpy as np


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


class RaiseEyebrows():
    def __init__(self, right_indices=RIGHT_EYE, left_indices=LEFT_EYE):
        self.right_indices = right_indices
        self.left_indices = left_indices
        self.left_eyebrow_corner_ind = 285
        self.right_eyebrow_corner_ind = 55

    def __call__(self, landmarks):
        landmarks = np.array([L['results'] for L in landmarks])

        # Right eyes
        # horizontal line
        rh_right = landmarks[self.right_indices[0]]
        rh_left = landmarks[self.right_indices[8]]
        # vertical line
        rv_top = landmarks[self.right_eyebrow_corner_ind]
        rv_bottom = landmarks[self.right_indices[8]]
        # draw lines on right eyes
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[self.left_indices[0]]
        lh_left = landmarks[self.left_indices[8]]

        # vertical line
        lv_top = landmarks[self.left_eyebrow_corner_ind]
        lv_bottom = landmarks[self.left_indices[0]]

        rh_distance = euclaideanDistance(rh_right, rh_left)
        rv_distance = euclaideanDistance(rv_top, rv_bottom)

        lv_distance = euclaideanDistance(lv_top, lv_bottom)
        lh_distance = euclaideanDistance(lh_right, lh_left)

        re_ratio = rh_distance / rv_distance
        le_ratio = lh_distance / lv_distance

        return -re_ratio, -le_ratio


class GesturePlaceHolder(BaseRuleGesture):
    def __init__(self):
        super().__init__(th=0)

    def __call__(self, landmarks):
        return [-100, -100]


class Blinking:
    def __init__(self, right_indices, left_indices):
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

        return re_ratio, le_ratio


class OpenLips:
    def __init__(self):
        self.left_lip_corner_ind = 308
        self.right_lip_corner_ind = 78
        self.upper_lip_ind = 0
        self.lower_lip_ind = 14
        self.right_indices = RIGHT_EYE
        self.left_indices = LEFT_EYE
        self.forehead_ind = 9
        self.nose_ind = 4

    def __call__(self, landmarks):
        landmarks = np.array([L['results'] for L in landmarks])

        # Right eyes
        # horizontal line of eyes - as reference
        rh_right = landmarks[self.right_indices[0]]
        rh_left = landmarks[self.right_indices[8]]

        lh_right = landmarks[self.left_indices[0]]
        lh_left = landmarks[self.left_indices[8]]

        # Nose
        # Vertical line for reference
        top_nose = landmarks[self.forehead_ind]
        bottom_nose = landmarks[self.nose_ind]

        # Vertical lips line
        top_lip = landmarks[self.upper_lip_ind]
        bottom_lip = landmarks[self.lower_lip_ind]

        # Horizontal lips line
        left_lip_corner_ind = landmarks[self.left_lip_corner_ind]
        right_lip_corner_ind = landmarks[self.right_lip_corner_ind]

        # Mean horizontal eye reference
        rh_distance = euclaideanDistance(rh_right, rh_left)
        lh_distance = euclaideanDistance(lh_right, lh_left)
        mean_eye_horizontal = (rh_distance + lh_distance) / 2

        # Nose distance
        nose_distance = euclaideanDistance(top_nose, bottom_nose)

        # Horizontal lips
        h_lips_distance = euclaideanDistance(left_lip_corner_ind, right_lip_corner_ind)
        v_lips_distance = euclaideanDistance(top_lip, bottom_lip)

        horizontal_ratio = h_lips_distance / mean_eye_horizontal
        vertical_ratio = v_lips_distance / nose_distance

        return horizontal_ratio, vertical_ratio


class SingleActionTracker():
    def __init__(self, name: list[str], action: List[List[bool]] = None, hold_period: int = 0, wait_period: int = 0):

        if action is None:
            action = [False, False]
        self.name = name

        self.action = action
        self.hold_period = hold_period
        self.wait_period = wait_period
        self.wait_counter = 0
        self.hold_counter = 0
        self.grace_counter = 0
        self.grace_period = 2

        self.flags = [False, False]
        self.state = False

    def reset_state(self):
        self.wait_counter = 0
        self.hold_counter = 0
        self.grace_counter = 0
        self.flags = [-100]
        self.state = False

    def __call__(self, results):

        # place holder
        if results is None:
            self.flags = [-100]
            self.state = False
            return self.state

        self.flags = results

        # if the results are true
        if np.all(np.array(self.flags) == np.array(self.action)):
            # check if they are true long enough, if not, then increment counter
            self.grace_counter = 0
            if self.hold_counter < self.hold_period:
                self.hold_counter += 1
            # if the results are true long enough, start wait counter
            else:
                self.wait_counter = self.wait_period
        # if the results are false check wait and hold counters
        else:
            # if the hold condition is met, check the wait condition
            if self.hold_counter >= self.hold_period:
                # if the wait condition is met, return true, if not, increment wait counter
                if self.wait_counter == 0:
                    self.state = True
                else:
                    self.wait_counter -= 1
            else:
                if self.grace_counter > self.grace_period:
                    self.reset_state()
                else:
                    self.grace_counter += 1
        return self.state

    def __str__(self):
        return (f'Gesture name: {self.name}, Actions: {self.action}, Flags: {self.flags}, State: {self.state},'
                f' Hold counter: {self.hold_counter}, Wait counter: {self.wait_counter}')


class GlobalGestureExtractor:
    def __init__(self, action_map: dict = None):

        self.action_map = None

        self.update_action_map(action_map)
        self.results = {n: [0, 0] for n in self.action_map.keys()}

    def update_action_map(self, action_map):
        default_action_map = {
            'eyes': Blinking(right_indices=RIGHT_EYE, left_indices=LEFT_EYE),
            'eyebrows': RaiseEyebrows(right_indices=RIGHT_EYE, left_indices=LEFT_EYE),
            'lips': OpenLips()
        }

        if action_map is not None:
            default_action_map.update(action_map)
        self.action_map = default_action_map
        return

    def __call__(self, lm):
        '''

        :param lm: land mark array
        :return flag:
        '''

        results = {}
        for name, func in self.action_map.items():
            results[name] = (func(lm))
        self.results = results
        return results


class CustomGesture:
    def __init__(self, name, command_array: str, threshold_map: dict = None, reset_period: int = 150, wait_period=0):
        '''

        :param name: A descriptive name of gesture sequence
        :param command_array: String array of commands
        :param threshold_map: Thresholds for each command
        :param reset_period: Number of frames to complete the sequence before resetting
        :param wait_period: Number of frames to prevent reset and block other command after a changing state to True
        '''
        self.threshold_map = dict()
        self.update_default_threshold_map(threshold_map)
        self.name = name
        self.reset_period = reset_period
        self.reset_counter = 0
        self.command_array_text = command_array

        self.available_parts_dict = {
            'right_eye': {'open': [False, False], 'close': [True, False]},
            'left_eye': {'open': [False, False], 'close': [False, True]},
            'both_eyes': {'open': [False, False], 'close': [True, True]},

            'right_eyebrow': {'down': [False, False], 'up': [True, False]},
            'left_eyebrow': {'down': [False, False], 'up': [False, True]},
            'both_eyebrows': {'down': [False, False], 'up': [True, True]},

            'lips_vertical': {'close': [False, False], 'open': [False, True]},
            'lips_horizontal': {'close': [False, False], 'open': [True, False]},
            'lips_wide': {'close': [False, False], 'open': [True, True]}
        }

        self.action_translation = {
            'right_eye': 'eyes',
            'left_eye': 'eyes',
            'both_eyes': 'eyes',

            'right_eyebrow': 'eyebrows',
            'left_eyebrow': 'eyebrows',
            'both_eyebrows': 'eyebrows',

            'lips_vertical': 'lips',
            'lips_horizontal': 'lips',
            'lips_wide': 'lips'}

        self.action_array = self.pars_command_array(command_array)
        self.default_reset_state_action_name = 'both_eyes'
        self.reset_state_action = self.get_default_reset_action()
        self.random_blink = self.get_random_blink()
        self.wait_counter = 0
        self.wait_period = wait_period
        self.state = False

        self.blinking_protection = True

    def update_default_threshold_map(self, threshold_map):
        default_threshold_map = {
            'right_eye': [5.5, 100],
            'left_eye': [100, 5.5],
            'both_eyes': [5.5, 5.5],

            'left_eyebrow': [10, -1.1],
            'right_eyebrow': [-1.1, 10],
            'both_eyebrows': [-1.1, -1.1],

            'lips_vertical': [10, 0.5],
            'lips_horizontal': [1.8, 10],
            'lips_wide': [1.8, 0.5]
        }

        if threshold_map is not None:
            default_threshold_map.update(threshold_map)
        self.threshold_map = default_threshold_map
        return

    def reset_state(self):
        '''

        :return:
        '''

        if self.wait_counter > 0:  # If in waiting period, do not reset
            return

        for action_object in self.action_array:
            action_object.reset_state()

        self.reset_counter = 0
        self.state = False

    def pars_command_array(self, command_array) -> List[SingleActionTracker]:
        '''
        A command array is a series of actions that represent a command.
        An action is a basic or combination of basic facial gestures performed at the same time.
        A facial gesture has four steps:
            step 1. The motion of part of the face from default position to a different position
            step 2. Holding the new position for a period of time (number of frames) - HOLD PERIOD
            step 3. Returning to default position
            step 4. Remaining at the default position for a period of time - WAIT PERIOD

        Example for a single action:
            step 1. Lifting the right eyebrow and closing the left eye
            step 2. Holding new position of both parts for a number of frames
            step 3. Lowering the right eyebrow and opening the left eye
            step 4. Keeping both parts at the default position for a number of frames

        The command array must have the following structure:
        [action1]->[action2]->[action3]...->[actionN]
        The different actions are separated by an arrow (->)
        while each action is comprised of:
            <part1>:<action1>&<part2>:<action2>|<number of frames>|<number of frames at base before next action>

        Each part (of the face) and motion are separated by colon (:),
        it is possible to add multiple part-motion pairs for a single action, separated by an ampersand (&).

        The first number following the (|) indicated the number of frames to hold the action
        The second number following the (|) indicated the number of frames to wait after reverting to base (up->down)


        The available parts and motions are as follows, the base/default states are marked with an astrix:
            left_eye: open*, close.
            right_eye: open*, close.
            both_eyes: open*, close.
            left_eyebrow: down*, up.
            right_eyebrow: down*:, up.
            both_eyebrows: down*, up.
            lips_vertical: close*, open.
            lips_horizontal: close*, open.
            lips_wide: close*, open.

        example:
            command array: 'right_eye:close&both_eyebrows:up|10|3->left_eye:close|10|3'
            action1: close right eye and lift both eyebrows for 10 frames, open the right eye and lower both eyebrows for 3 frames
            action2: close left eye, keep closed for 10 frames, open the left eye and keep open for 3 frames


        :param command_array: a string of command
        :return single_action_tracker_array: a list of SingleActionTracker objects
        '''
        all_commands = []
        single_action_tracker_array = []
        command_array = command_array.split('->')
        # command_array = [c.split('&') for c in command_array]  # split each command into simultaneous commands

        for action in command_array:
            single_action_tracker_array.append(self.generate_single_action_tracker(action))
        return single_action_tracker_array

    def generate_single_action_tracker(self, action):
        sub_actions = action.split('|')
        assert len(sub_actions) == 3, f'Incorrect action, the action must contain 3 parts,' \
                                      f' but it only had {len(sub_actions)}, action: {action}'

        parts_actions_all = sub_actions[0]  # face part and gesture
        names = []
        flags = []
        parts_actions_list = parts_actions_all.split('&')
        for single_part_action in parts_actions_list:
            single_part_action_split = single_part_action.split(':')
            assert len(single_part_action_split) == 2, f'Bad format, this must be a facial part:gesture,' \
                                                       f' but it was {single_part_action} instead'
            face_part = single_part_action_split[0]
            face_gesture = single_part_action_split[1]
            flags.append(self.available_parts_dict[face_part][face_gesture])
            names.append(face_part)
        hold_period = sub_actions[1]
        wait_period = sub_actions[2]

        assert hold_period.isnumeric(), f'hold period must be a positive integer, but is was {hold_period}'
        assert wait_period.isnumeric(), f'wait_period period must be a positive integer, but is was {wait_period}'

        return SingleActionTracker(names, action=flags, hold_period=int(hold_period), wait_period=int(wait_period))

    def get_default_reset_action(self):
        # close both eyes for 3 frames
        if self.default_reset_state_action_name == 'none':
            return SingleActionTracker(name=['place holder'])

        return SingleActionTracker(name=[self.default_reset_state_action_name],
                                   action=[self.available_parts_dict['both_eyes']['close']],
                                   hold_period=20,
                                   wait_period=0)

    def get_random_blink(self):
        # close both eyes for 3 frames
        reset_state = SingleActionTracker(name=[self.default_reset_state_action_name],
                                          action=[self.available_parts_dict['both_eyes']['close']],
                                          hold_period=1,
                                          wait_period=0)
        return reset_state

    def get_all_states(self):
        ret_list = []
        for action in self.action_array:
            ret_list.append(np.all(action.state))
        return ret_list

    def call_sim_action(self, results, action_obj_array: list[SingleActionTracker]):
        return np.all([self.call_action(results, act) for act in action_obj_array])

    def persistent_action(self, results, action_name):
        action_name_translated = self.action_translation[action_name]
        res = results[action_name]
        if len(res) == 1:
            res = [res]

        assert (len(self.threshold_map[action_name])) == len(
            res), 'The length of the results and the length of the thresholds must match'
        res = np.array(res) > np.array(self.threshold_map[action_name])
        return np.all(res)

    def call_action(self, results, action_obj: SingleActionTracker):
        res = []
        for action_name in action_obj.name:
            res_temp = results[self.action_translation[action_name]]
            th_temp = self.threshold_map[action_name]
            res.append(np.array(res_temp) > np.array(th_temp))

        # if res[1]:
        #     a = 1
        return action_obj(res)

    def __call__(self, results):
        '''
        Update the single action trackers and check if all the tracker conditions are met.
        :param lm: land mark array
        :return flag: If all actions in the action array were performed, if not, then return false
        '''
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return True
        #
        # # check if reset command was issued
        if self.call_action(results, self.reset_state_action):
            pass
        #     self.reset_state_action.reset_state()
        #     self.random_blink.reset_state()
        #     self.reset_state()
        #     # print('RESET DETECTED')
        #     return False

        elif self.call_action(results, self.random_blink):  # ignore random blinking
            self.reset_state_action.reset_state()
            self.random_blink.reset_state()
            # print('BLINKING DETECTED')
            return False

        # if the first action in the list is active, increment the reset counter
        all_states = self.get_all_states()
        if all_states[0]:
            self.reset_counter += 1

        if self.reset_counter > self.reset_period:
            self.reset_state()

        for action, states in zip(self.action_array, all_states):
            # check if the first action flag
            if not states:
                ret = self.call_action(results, action)
                # ret = self.call_sim_action(results, action)
                if not ret:
                    break
            else:
                continue

        results_out = np.all(self.get_all_states())
        if results_out:
            if not self.state:  # This is the first activation, start waiting period
                self.state = True
                self.wait_counter = self.wait_period
        else:
            if self.state:  # This was activated and the waiting period is over, now reset
                self.reset_state()

        return results_out

    def proclaim_detection(self, print_flag=False):
        ret = f'{self.name}'
        if print_flag:
            print(ret)
        return ret

    def __str__(self):
        ret = f'Command name: {self.name}\n'
        for action in self.action_array:
            ret += f'{action}\n'
        return ret
