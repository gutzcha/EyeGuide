import cv2
import mediapipe as mp

import time

from demo.rules import GlobalGestureExtractor, CustomGesture
from demo.set_colors import get_custom_face_mesh_contours_style, get_facemesh_contours_connection_style
import numpy as np
from utils.constants import COLORS


class ReadGestures():
    def __init__(self, custom_gesture_array: list[CustomGesture], window_height=480, window_width=640, rescale_ratio=1):

        # display window formatting
        self.window_height = window_height
        self.window_width = window_width
        self.rescale_ratio = rescale_ratio
        self.final_height = int(self.window_height * rescale_ratio)
        self.final_width = int(self.window_width * rescale_ratio)

        # mediapipe pose extractor utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # action extractor
        self.global_extractor = GlobalGestureExtractor()

        # gesture recognizer
        self.custom_gesture_array = custom_gesture_array

        # color schema
        self.color_schema = self.get_default_colors()

    def run(self):
        cap = cv2.VideoCapture(0)

        n_frames = 0
        draw_mesh = True
        mp_face_mesh = self.mp_face_mesh
        WIDTH, HEIGHT = self.final_width, self.final_height

        results_facial_exp = None
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():

                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")

                    break

                image = cv2.resize(image, (WIDTH, HEIGHT))

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    if draw_mesh:
                        facemesh_contours_connection_style = self.set_colors_from_results(results_facial_exp)

                        for face_landmarks in results.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                    .get_default_face_mesh_tesselation_style())

                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=get_custom_face_mesh_contours_style(
                                    facemesh_contours_connection_style))

                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                    .get_default_face_mesh_iris_connections_style())

                else:
                    continue

                ################################################################################################################
                lm = self.reformat_landmarks(face_landmarks)
                results_facial_exp = self.global_extractor(lm)
                flags = [custom_gesture(results_facial_exp) for custom_gesture in self.custom_gesture_array]

                # debug
                # print(flags)
                # print(results_facial_exp)

                self.display_results(image, flags)
                self.reset_all(flags)
                # ###############################################################################################################

                cv2.imshow('MediaPipe Face Mesh', image)
                key = cv2.waitKey(1)
                # key = -1
                n_frames += 1

                if key == ord('q'):
                    break

        cap.release()

    def get_default_colors(self) -> dict:
        col_pos = COLORS['RED']
        col_neg = COLORS['BLUE']

        col_neg_face = COLORS['WHITE']
        col_lips_horizontal_pos = COLORS['GREEN']
        col_lips_vertical_pos = COLORS['YELLOW']
        col_lips_wide_pos = COLORS['PURPLE']

        color_schema = {
            'left_eye': {'pos': col_pos, 'neg': col_neg},
            'right_eye': {'pos': col_pos, 'neg': col_neg},
            'left_eyebrow': {'pos': col_pos, 'neg': col_neg},
            'right_eyebrow': {'pos': col_pos, 'neg': col_neg},
            'lips_vertical': {'pos': col_lips_vertical_pos, 'neg': col_neg_face},
            'lips_horizontal': {'pos': col_lips_horizontal_pos, 'neg': col_neg_face},
            'lips_wide': {'pos': col_lips_wide_pos, 'neg': col_neg_face},
            'face': {'pos': col_neg_face, 'neg': col_neg_face}
        }
        return color_schema

    def set_colors_from_results(self, res):
        custom_gesture_array = self.custom_gesture_array
        color_schema = self.color_schema

        threshold_map = custom_gesture_array[0].threshold_map
        if res is None:
            return get_facemesh_contours_connection_style(
                color_left_eye=color_schema['left_eye']['neg'],
                color_right_eye=color_schema['right_eye']['neg'],
                color_left_eyebrow=color_schema['left_eyebrow']['neg'],
                color_right_eyebrow=color_schema['right_eyebrow']['neg'],
                color_lips=color_schema['lips_vertical']['neg'],
                color_face=color_schema['face']['neg'])

        # left eye
        if res['eyes'][1] > threshold_map['left_eye'][1]:
            col_left_eye = color_schema['left_eye']['pos']
        else:
            col_left_eye = color_schema['left_eye']['neg']

        # right eye
        if res['eyes'][0] > threshold_map['right_eye'][0]:
            col_right_eye = color_schema['right_eye']['pos']
        else:
            col_right_eye = color_schema['right_eye']['neg']

        # left eyebrow
        if res['eyebrows'][1] > threshold_map['left_eyebrow'][1]:
            col_left_eyebrow = color_schema['left_eyebrow']['pos']
        else:
            col_left_eyebrow = color_schema['left_eyebrow']['neg']

        # right eyebrow
        if res['eyebrows'][0] > threshold_map['right_eyebrow'][0]:
            col_right_eyebrow = color_schema['right_eyebrow']['pos']
        else:
            col_right_eyebrow = color_schema['right_eyebrow']['neg']

        # lips horizontal or vertical or wide
        if res['lips'][0] > threshold_map['lips_horizontal'][0] and res['lips'][1] > threshold_map['lips_vertical'][1]:
            col_lips = color_schema['lips_wide']['pos']
        elif res['lips'][0] > threshold_map['lips_horizontal'][0]:
            col_lips = color_schema['lips_horizontal']['pos']
        elif res['lips'][1] > threshold_map['lips_vertical'][1]:
            col_lips = color_schema['lips_vertical']['pos']
        else:
            col_lips = color_schema['face']['neg']

        return get_facemesh_contours_connection_style(
            color_left_eye=col_left_eye,
            color_right_eye=col_right_eye,
            color_left_eyebrow=col_left_eyebrow,
            color_right_eyebrow=col_right_eyebrow,
            color_lips=col_lips,
            color_face=color_schema['face']['neg'])

    def reformat_landmarks(self, face_landmarks):
        all_landmarks_x = [l.x for l in face_landmarks.landmark]
        all_landmarks_y = [l.y for l in face_landmarks.landmark]
        # all_landmarks_z = [l.z for l in face_landmarks.landmark]
        res = []
        for i, (x, y) in enumerate(zip(all_landmarks_x, all_landmarks_y)):
            # print(f'{i}:{(x,y,z)}')
            res.append(dict(results=[x, y]))
        return res

    def display_results(self, image, flags):
        height, width, _ = image.shape
        for flag, custom_gesture in zip(flags, self.custom_gesture_array):
            if flag:
                print_string = custom_gesture.proclaim_detection()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(image, (int(width * 0.05), int(height * 0.8)),
                              (int(width * 0.9), int(height * 0.98)),
                              COLORS['WHITE'], -1)
                cv2.putText(image, print_string, (int(width * 0.05), int(height * 0.9)), font, 0.7,
                            COLORS['RED'], 2,
                            cv2.LINE_AA)
        return

    def reset_all(self, flags):
        for flag in flags:
            if flag:
                [c.reset_state() for c in self.custom_gesture_array]


class FacialWriting(ReadGestures):
    def __init__(self):
        self.current_string = ''
        get_gesture_array = self.get_gesture_array()
        super(FacialWriting, self).__init__(get_gesture_array)
        self.caps_on = False

    def change_caps_lock(self):
        self.caps_on = not self.caps_on

    def display_results(self, image, flags):
        height, width, _ = image.shape
        for flag, custom_gesture in zip(flags, self.custom_gesture_array):
            if flag:

                print_string = custom_gesture.proclaim_detection()

                # debug
                # print(print_string)
                if print_string == 'back':
                    if len(self.current_string) > 0:
                        self.current_string = self.current_string[:-1]
                elif print_string == 'capslock':
                    self.change_caps_lock()
                else:
                    print_string = print_string.lower() if self.caps_on else print_string.upper()
                    self.current_string = self.current_string + print_string
                self.reset_all(flags)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(image, (int(width * 0.05), int(height * 0.8)),
                      (int(width * 0.9), int(height * 0.98)),
                      COLORS['WHITE'], -1)
        cv2.putText(image, self.current_string, (int(width * 0.05), int(height * 0.9)), font, 0.7,
                    COLORS['RED'], 2,
                    cv2.LINE_AA)
        if self.caps_on:
            cv2.putText(image, 'Caps Lock On', (int(width * 0.05), int(height * 0.85)), font, 0.7,
                        COLORS['BLUE'], 1,
                        cv2.LINE_AA)

        return

    def get_gesture_array(self):
        gesture_letter_array = [

            CustomGesture('A', 'right_eye:close|5|3->right_eye:close|5|15', reset_period=120),
            CustomGesture('B', 'right_eye:close|5|3->left_eye:close|5|15', reset_period=120),
            CustomGesture('C', 'both_eyebrows:up|5|3->both_eyebrows:up|5|15', reset_period=120),
            CustomGesture('D', 'lips_vertical:open|10|3->lips_vertical:open|10|15', reset_period=120),
            CustomGesture('E', 'lips_vertical:open|10|3->lips_horizontal:open|10|15', reset_period=120),
            CustomGesture('F', 'lips_horizontal:open|10|3->lips_horizontal:open|10|15', reset_period=120),

            CustomGesture('capslock', 'both_eyebrows:up|30|3', reset_period=120),
            CustomGesture('back', 'both_eyes:close|10|3', reset_period=120)

        ]
        return gesture_letter_array


if __name__ == '__main__':
    command_array_1 = 'right_eye:close&both_eyebrows:up|10|3->left_eye:close|10|3'
    command_array_2 = 'both_eyebrows:up|10|3->both_eyebrows:up|10|3->both_eyebrows:up|10|3'
    command_array_3 = 'lips_vertical:open|10|3->lips_horizontal:open|10|3->lips_wide:open|10|3'
    # command_array_4 = 'q:up|10|3->right_eye:close|10|3->lips_wide:open|10|3'
    command_array_5 = 'lips_vertical:open|10|3->lips_vertical:open|10|50'
    command_array_6 = 'right_eye:close|10|3->right_eye:close|10|3->right_eye:close|10|3'
    command_array_7 = 'lips_vertical:open|10|3->lips_vertical:open|10|3->lips_vertical:open|10|3'

    default_command_array_1 = 'both_eyes:close|30|3'
    default_command_array_2 = 'both_eyebrows:up|20|60'
    default_command_array_3 = 'both_eyebrows:up|20|3->both_eyebrows:up|20|30'

    custom_gesture_array = [
        CustomGesture('Take a screenshot: Wink right + both eyebrows, left', command_array_1, wait_period=100),
        CustomGesture('Left click and hold: Lift both eyebrowsX3', command_array_2, wait_period=100),
        CustomGesture('Zoom in/out mode: Smile, say ahh, lips wide', command_array_3, wait_period=100),
        # CustomGesture('Dragging mode: Lift both eyebrows, right eye, lips wide', command_array_4, wait_period=100),
        CustomGesture('Left click: Say ahh X2', command_array_5, wait_period=100),
        CustomGesture('Close window: Right wink X3', command_array_6, wait_period=100),
        CustomGesture('Left click and hold: Say ahh X3', command_array_7, wait_period=100),
        CustomGesture('Reset all gestures: close both eyes for 2 sec', default_command_array_1, wait_period=100),
        CustomGesture('Right Click: Rise both eyebrows for 2 sec', default_command_array_2, wait_period=100),
        CustomGesture('Double Click: Rise both eyebrows for 2 sec X2 ', default_command_array_3, wait_period=100)
    ]
    facial_writing = FacialWriting()
    facial_writing.run()

    # read_gesture = ReadGestures(custom_gesture_array)
    # read_gesture.run()
