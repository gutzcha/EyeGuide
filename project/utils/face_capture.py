import cv2
import mediapipe as mp
from project.utils.constants import gestures_dict
import pickle
import time
from constants import important_keypoints, inv_eye_keypoints
import os.path as osp
from demo.rules import TrippelWink


rescale_ratio = 1
HEIGHT, WIDTH = int(480 * rescale_ratio), int(640 * rescale_ratio)
triple_wink = TrippelWink()
blink_det = triple_wink.blink_detector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
keypoints = list(mp_face_mesh.FACEMESH_IRISES)

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


def spatial_normalization(landmarks):
    pass


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


# For static images:
# IMAGE_FILES = []
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5) as face_mesh:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB before processing.
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     # Print and draw face mesh landmarks on the image.
#     if not results.multi_face_landmarks:
#       continue
#     annotated_image = image.copy()
#     for face_landmarks in results.multi_face_landmarks:
#       print('face_landmarks:', face_landmarks)
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_TESSELATION,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_tesselation_style())
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_CONTOURS,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_contours_style())
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_iris_connections_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
root_folder = 'C:\\Users\\user\\EyeGuide\\data'
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
video_path = '300VW_Dataset_2015_12_14\\300VW_Dataset_2015_12_14\\001\\vid.avi'
# cap = cv2.VideoCapture(osp.join(root_folder, video_path))
cap = cv2.VideoCapture(0)
flag_counter = 0
# q - quit
# w - blink left
# e - blink right
# r - smile
# t - frown
#   - gb


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
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())

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

        # face_landmarks_arr.append(results.multi_face_landmarks[0])
        # img_arr.append(image)
        # image_orig_arr.append(image_orig)

        # [draw_landmark(face_landmarks, ind=i, img=image, text_flag=True) for i in range(len(face_landmarks.landmark)) if i%8==0]
        draw_mini_face(face_landmarks, image)

        # Flip the image horizontally for a selfie-view display.
        # image = cv2.flip(image, 1)

        # display fps

        fps, prev_frame_time = get_fps(prev_frame_time)

        if n_frames % 5 == 0:
            last_fps = fps

        display_fps(last_fps, image)

        cv2.imshow('MediaPipe Face Mesh', image)
        key = cv2.waitKey(1)
        # key = -1
        n_frames += 1
        # if key == -1:
            # gesture_arr.append('bg')
        # elif chr(key) in gestures_dict:
            # gesture_arr.append(gestures_dict[chr(key)])
            # print(chr(key))
        # elif key == ord('q'):
        if key == ord('q'):
            break


        # test wink detection
        lm = reformat_landmarks(face_landmarks)

        trip_flag = triple_wink(lm)
        # rb, lb = blink_det(lm)
        # print(f'Right eye blink:{triple_wink.wink_r}, Left eye blink:{triple_wink.wink_l}')
        if trip_flag:
            flag_counter = 1
        if flag_counter > 0:
            flag_counter -= 1
            print(f'Triple Blink Detected:{n_frames}')


cap.release()
all_frames = {}
save_flag = False
if save_flag:
    for label, landmark, image_orig, frame_landmark, frame_id in zip(gesture_arr, face_landmarks_arr, image_orig_arr,
                                                                     img_arr, range(n_frames)):
        all_frames[frame_id] = {'label': label,
                                'landmarks': landmark,
                                'frame_orig': image_orig,
                                'frame_landmarks': frame_landmark}
    # Directly from dictionary
    timestr = time.strftime("%Y%m%d-%H%M")
    with open(f'../assets/dataset_{timestr}.pickle', 'wb') as handle:
        pickle.dump(all_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
