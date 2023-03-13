import cv2
import mediapipe as mp
from project.utils.constants import gestures_dict
import pickle
import time
from constants import important_keypoints, inv_eye_keypoints
import os.path as osp
from demo.rules import TrippelWink, RaiseEyebrows, GlobalGestureExtractor, CustomGesture
from demo.set_colors import get_custom_face_mesh_contours_style, get_facemesh_contours_connection_style
from mediapipe.python.solutions import face_mesh_connections
import numpy as np

rescale_ratio = 1
HEIGHT, WIDTH = int(480 * rescale_ratio), int(640 * rescale_ratio)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

keypoints = list(mp_face_mesh.FACEMESH_IRISES)
keypoints_eyebrows_left = face_mesh_connections.FACEMESH_LEFT_EYEBROW
keypoints_eyebrows_right = face_mesh_connections.FACEMESH_RIGHT_EYEBROW

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

col_pos = _RED
col_neg = _BLUE

col_left_eye = col_neg
col_right_eye = col_neg
col_left_eyebrow = col_neg
col_right_eyebrow = col_neg
col_face = _WHITE
col_lips = _WHITE

command_array_1 = 'right_eye|close|20|15->left_eye|close|20|15->right_eye|close|20|15'
command_array_2 = 'both_eyebrows|up|10|3->both_eyebrows|up|10|3->both_eyebrows|up|10|3'
command_array_3 = 'lips_vertical|open|10|3->lips_horizontal|open|10|3->lips_wide|open|10|3'
command_array_4 = 'both_eyebrows|up|10|3->right_eye|close|10|3->lips_wide|open|10|3'
command_array_5 = 'lips_vertical|open|10|3->lips_vertical|open|10|50'
command_array_6 = 'right_eye|close|10|3->right_eye|close|10|3->right_eye|close|10|3'
command_array_7 = 'lips_vertical|open|10|3->lips_vertical|open|10|3->lips_vertical|open|10|3'

default_command_array_1 = 'both_eyes|close|30|3'
default_command_array_2 = 'both_eyebrows|up|20|30'
default_command_array_3 = 'both_eyebrows|up|20|3->both_eyebrows|up|20|30'


custom_gesture_array = [
    CustomGesture('Take a screenshot: Wink right, left right', command_array_1),
    CustomGesture('Left click and hold: Lift both eyebrowsX3', command_array_2),
    CustomGesture('Zoom in/out mode: Smile, say ahh, lips wide', command_array_3),
    CustomGesture('Dragging mode: Lift both eyebrows, right eye, lips wide', command_array_4),
    CustomGesture('Left click: Say ahh X2', command_array_5),
    CustomGesture('Close window: Right wink X3', command_array_6),
    CustomGesture('Left click and hold: Say ahh X3', command_array_7),
    CustomGesture('Reset all gestures: close both eyes for 2 sec', default_command_array_1),
    CustomGesture('Right Click: Rise both eyebrows for 2 sec', default_command_array_2),
    CustomGesture('Double Click: Rise both eyebrows for 2 sec X2 ', default_command_array_3)
]


# custom_gesture_array = [
#     CustomGesture('Eyes: right, left right', command_array_1),
# ]

global_extractor = GlobalGestureExtractor()


def set_colors_from_results(res):
    threshold_map = custom_gesture_array[0].threshold_map
    if res is None:
        return get_facemesh_contours_connection_style(
            color_left_eye=col_neg,
            color_right_eye=col_neg,
            color_left_eyebrow=col_neg,
            color_right_eyebrow=col_neg,
            color_lips=col_neg,
            color_face=col_neg)

    # left eye
    if res['eyes'][1] > threshold_map['left_eye'][1]:
        col_left_eye = col_pos
    else:
        col_left_eye = col_neg

    # right eye
    if res['eyes'][0] > threshold_map['right_eye'][0]:
        col_right_eye = col_pos
    else:
        col_right_eye = col_neg

    # left eyebrow
    if res['eyebrows'][1] > threshold_map['left_eyebrow'][1]:
        col_left_eyebrow = col_pos
    else:
        col_left_eyebrow = col_neg

    # right eyebrow
    if res['eyebrows'][0] > threshold_map['right_eyebrow'][0]:
        col_right_eyebrow = col_pos
    else:
        col_right_eyebrow = col_neg

    # lips horizontal or vertical or wide
    if res['lips'][0] > threshold_map['lips_horizontal'][0] and res['lips'][1] > threshold_map['lips_vertical'][1]:
        col_lips = _PURPLE
    elif res['lips'][0] > threshold_map['lips_horizontal'][0]:
        col_lips = _GREEN
    elif res['lips'][1] > threshold_map['lips_vertical'][1]:
        col_lips = _YELLOW
    else:
        col_lips = col_neg

    return get_facemesh_contours_connection_style(
        color_left_eye=col_left_eye,
        color_right_eye=col_right_eye,
        color_left_eyebrow=col_left_eyebrow,
        color_right_eyebrow=col_right_eyebrow,
        color_lips=col_lips,
        color_face=col_face)


def reformat_landmarks(face_landmarks):
    all_landmarks_x = [l.x for l in face_landmarks.landmark]
    all_landmarks_y = [l.y for l in face_landmarks.landmark]
    # all_landmarks_z = [l.z for l in face_landmarks.landmark]
    res = []
    for i, (x, y) in enumerate(zip(all_landmarks_x, all_landmarks_y)):
        # print(f'{i}:{(x,y,z)}')
        res.append(dict(results=[x, y]))
    return res


def draw_eyes(face_landmarks, img, landmark_inds):
    if len(landmark_inds) < 2:
        landmark_inds = list(landmark_inds)
    for i in landmark_inds:
        these_face_landmarks = [face_landmarks.landmark[x] for x in keypoints[i]]
        for landmak in these_face_landmarks:
            height, width, c = image.shape
            x = int(landmak.x * width)
            y = int(landmak.y * height)
            img = cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
    return image


def display_fps(fps, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # putting the FPS count on the frame
    cv2.putText(img, fps, (int(WIDTH * 0.8), 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)


def get_fps(prev_frame_time):
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    return str(fps), new_frame_time


def draw_landmark(face_landmarks, ind, img, text_flag=False):
    landmak = face_landmarks.landmark[ind]
    height, width, c = image.shape
    x = int(landmak.x * width)
    y = int(landmak.y * height)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if text_flag:
        img = cv2.putText(img, str(ind), (x, y), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    else:
        img = cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
    return img


def draw_mini_face(face_landmarks, img):
    mini_win_scale = 0.2
    offset_x = 0.01 * WIDTH
    offset_y = 0.1 * HEIGHT

    all_landmarks_x = [l.x for l in face_landmarks.landmark]
    all_landmarks_y = [l.y for l in face_landmarks.landmark]
    all_landmarks_z = [l.z for l in face_landmarks.landmark]
    # res = []
    # for i, (x,y,z) in enumerate(zip(all_landmarks_x,all_landmarks_y,all_landmarks_z)):
    #     # print(f'{i}:{(x,y,z)}')
    #     res.append(dict(results=[x,y]))
    min_x, max_x, diff_x = min(all_landmarks_x), max(all_landmarks_x), max(all_landmarks_x) - min(all_landmarks_x)
    min_y, max_y, diff_y = min(all_landmarks_y), max(all_landmarks_y), max(all_landmarks_y) - min(all_landmarks_y)

    start_point = (int(offset_x), int(offset_y))
    end_point = (int(WIDTH * mini_win_scale + offset_x), int(HEIGHT * mini_win_scale + offset_y))

    cv2.rectangle(img, start_point, end_point, (250, 250, 250), -1)
    for lx, ly in zip(all_landmarks_x, all_landmarks_y):
        lx = int((lx - min_x) / diff_x * WIDTH * mini_win_scale + offset_x)
        ly = int((ly - min_y) / diff_y * HEIGHT * mini_win_scale + offset_y)

        cv2.circle(img, (lx, ly), 1, (255, 0, 255), -1)


# For webcam input:
root_folder = 'C:\\Users\\user\\EyeGuide\\data'
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
video_path = '300VW_Dataset_2015_12_14\\300VW_Dataset_2015_12_14\\001\\vid.avi'
# cap = cv2.VideoCapture(osp.join(root_folder, video_path))
cap = cv2.VideoCapture(0)
flag_counter = 0

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

gesture_arr = []
face_landmarks_arr = []
img_arr = []
image_orig_arr = []
n_frames = 0
all_frames = {}
# draw_mesh = False
draw_mesh = True
blink_state = False
blink_counter = 0
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
            # If loading a video, use 'break' instead of 'continue'.
            # continue
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
        image_orig = image.copy()

        if results.multi_face_landmarks:
            if draw_mesh:
                facemesh_contours_connection_style = set_colors_from_results(results_facial_exp)
                # facemesh_contours_connection_style = get_facemesh_contours_connection_style(
                #     color_left_eye=col_left_eye,
                #     color_right_eye=col_right_eye,
                #     color_left_eyebrow=col_left_eyebrow,
                #     color_right_eyebrow=col_right_eyebrow,
                #     color_lips=col_lips,
                #     color_face=col_face)

                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=get_custom_face_mesh_contours_style(facemesh_contours_connection_style))

                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #         .get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

                    # image = draw_eyes(face_landmarks, image, landmark_inds= [0])
        else:
            continue

        keypoints_eyebrows_left_list = list(set(np.array(list(keypoints_eyebrows_left)).ravel()))
        keypoints_eyebrows_right_list = list(set(np.array(list(keypoints_eyebrows_right)).ravel()))

        # [draw_landmark(face_landmarks, ind=i, img=image, text_flag=True) for i in keypoints_eyebrows_right_list]
        draw_mini_face(face_landmarks, image)

        # Flip the image horizontally for a selfie-view display.
        # image = cv2.flip(image, 1)

        # display fps
        fps, prev_frame_time = get_fps(prev_frame_time)

        if n_frames % 5 == 0:
            last_fps = fps

        display_fps(last_fps, image)


        ################################################################################################################
        lm = reformat_landmarks(face_landmarks)
        results_facial_exp = global_extractor(lm)
        flags = [custom_gesture(results_facial_exp) for custom_gesture in custom_gesture_array]
        # default_flags = [default_gesture(results_facial_exp) for default_gesture in default_gesture_array]

        # print(custom_gesture)
        # print(custom_gesture_array[0])
        # print(results_facial_exp['lips'])
        # for flag, default_gesture in zip(default_flags, default_gesture_array):
        #     if flag:
        #         print_string = default_gesture.proclaim_detection()
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cv2.rectangle(image, (int(WIDTH * 0.05), int(HEIGHT * 0.8)), (int(WIDTH * 0.9), int(HEIGHT * 0.98)),
        #                       _WHITE, -1)
        #         cv2.putText(image, print_string, (int(WIDTH * 0.05), int(HEIGHT * 0.9)), font, 0.7, _RED, 2,
        #                     cv2.LINE_AA)
        #         # reset everything else
        #         [c.reset_state() for c in custom_gesture_array]
        #         [c.reset_state() for c in default_gesture_array]

        for flag, custom_gesture in zip(flags, custom_gesture_array):
            if flag:
                print_string = custom_gesture.proclaim_detection()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(image, (int(WIDTH*0.05), int(HEIGHT*0.8)), (int(WIDTH*0.9), int(HEIGHT*0.98)), _WHITE, -1)
                cv2.putText(image, print_string, (int(WIDTH*0.05), int(HEIGHT*0.9)), font, 0.7, _RED, 2, cv2.LINE_AA)
                # reset everything else
                [c.reset_state() for c in custom_gesture_array]
        ################################################################################################################

        cv2.imshow('MediaPipe Face Mesh', image)
        key = cv2.waitKey(1)
        # key = -1
        n_frames += 1

        if key == ord('q'):
            break

cap.release()
all_frames = {}
