import cv2
import mediapipe as mp





class MediapipeVideoExtractor:
    def __init__(self, config_all,  debug=False):

        self.debug = debug

        # set video properties
        config_video = config_all['config_video']
        self.img_width = config_video['img_width']
        self.img_height = config_video['img_height']
        self.outputs_width = config_video['output_width']
        self.outputs_height = config_video['output_height']
        self.video_path = config_video['video_path']

        # set display options
        config_display = config_all['config_display']
        self.draw_landmarks_flag = config_display['draw_landmarks_flag']
        self.draw_mini_face_flag = config_display['draw_mini_face_flag']
        self.draw_video_flag = config_display['draw_video_flag']

        # init mediapipe and opencv objects
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils
        self.cv, self.fps = self.init_cv()

        # init results
        self.image = None
        self.results = []

    def init_cv(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cap, fps

    def draw_mini_face(self, img, face_landmarks):
        if not face_landmarks:
            return []

        draw_flag = self.draw_mini_face_flag
        normalized_results = []
        width = self.img_width
        height = self.img_height

        mini_win_scale = 0.2

        offset_x = 0.01 * width
        offset_y = 0.1 * height

        all_landmarks_x = [l.x for l in face_landmarks.landmark]
        all_landmarks_y = [l.y for l in face_landmarks.landmark]
        min_x, max_x, diff_x = min(all_landmarks_x), max(all_landmarks_x), max(all_landmarks_x) - min(all_landmarks_x)
        min_y, max_y, diff_y = min(all_landmarks_y), max(all_landmarks_y), max(all_landmarks_y) - min(all_landmarks_y)

        start_point = (int(offset_x), int(offset_y))
        end_point = (int(width * mini_win_scale + offset_x), int(height * mini_win_scale + offset_y))

        for lx, ly in zip(all_landmarks_x, all_landmarks_y):
            lx_norm = (lx - min_x) / diff_x
            ly_norm = (ly - min_y) / diff_y
            normalized_results.append((lx_norm, ly_norm))

            # scale the mini face and move it from the edge before drawing
        if draw_flag:
            cv2.rectangle(img, start_point, end_point, (250, 250, 250), -1)
            for (lx_norm, ly_norm) in normalized_results:
                lx_plot = int(lx_norm * width * mini_win_scale + offset_x)
                ly_plot = int(ly_norm * height * mini_win_scale + offset_y)
                cv2.circle(img, (lx_plot, ly_plot), 1, (255, 0, 255), -1)

        return normalized_results

    def draw_face_landmarks(self, image, face_landmarks):

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True

        mp_face_mesh = self.mp_face_mesh
        mp_drawing_styles = self.mp_drawing_styles

        if face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

    def get_raw_face_landmarks(self,face_landmarks):
        all_landmarks_x = [int(l.x*self.img_width) for l in face_landmarks.landmark]
        all_landmarks_y = [int(l.y*self.img_height) for l in face_landmarks.landmark]
        return [(x, y) for x, y in zip(all_landmarks_x, all_landmarks_y)]
    def process(self):
        cap = self.cv
        fps = self.fps

        img_width, img_height = self.img_width, self.img_height
        frame_ind = 0
        with self.mp_face_mesh.FaceMesh(
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

                image = cv2.resize(image, (img_width, img_height))
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


                temp_results = face_mesh.process(image)
                if temp_results:
                    face_landmarks = temp_results.multi_face_landmarks[0]
                else:
                    face_landmarks = None


                timestamp = (cap.get(cv2.CAP_PROP_POS_MSEC))

                if self.debug:

                    self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    normalized_face = self.draw_mini_face(self.image, face_landmarks)
                    self.draw_face_landmarks(self.image, face_landmarks)

                    cv2.imshow('MediaPipe Face Mesh', self.image)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                    self.results += [
                        {'id': frame_ind,
                         # 'image': self.image.tolist(),
                         'image': self.image,
                         'results': self.get_raw_face_landmarks(face_landmarks),
                         'normalized_face': normalized_face,
                         'timestamps': timestamp}]
                else:
                    self.results += [
                        {'id': frame_ind,
                         'results': self.get_raw_face_landmarks(face_landmarks),
                         'timestamps': timestamp}]


                frame_ind += 1
        cap.release()
        return self.results
