import cv2
import mediapipe as mp


class MediapipeVideoObj():
    def __init__(self, config):
        self.img_width = config['img_width']
        self.img_height = config['img_height']
        self.outputs_width = config['output_width']
        self.outputs_height = config['output_height']
        self.video_path = config['video_path']

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils
        self.image = None
        self.draw_landmarks_flag = True
        self.draw_mini_face_flag = True
        self.draw_video_flag = True

        self.results = []



        self.init_cv()

    def init_cv(self):
        self.cv = cv2.VideoCapture(self.video_path)

    def draw_mini_face(self, face_landmarks, img):

        width = self.img_width
        height = self.img_height

        mini_win_scale = 0.2

        normalized_width = 1
        normalized_height = 1

        offset_x = 0.01 * width
        offset_y = 0.1 * height

        all_landmarks_x = [l.x for l in face_landmarks.landmark]
        all_landmarks_y = [l.y for l in face_landmarks.landmark]
        min_x, max_x, diff_x = min(all_landmarks_x), max(all_landmarks_x), max(all_landmarks_x) - min(all_landmarks_x)
        min_y, max_y, diff_y = min(all_landmarks_y), max(all_landmarks_y), max(all_landmarks_y) - min(all_landmarks_y)

        start_point = (int(offset_x), int(offset_y))
        end_point = (int(width * mini_win_scale + offset_x), int(height * mini_win_scale + offset_y))

        cv2.rectangle(img, start_point, end_point, (250, 250, 250), -1)
        for lx, ly in zip(all_landmarks_x, all_landmarks_y):
            lx = int((lx - min_x) / diff_x * width * mini_win_scale + offset_x)
            ly = int((ly - min_y) / diff_y * height * mini_win_scale + offset_y)

            cv2.circle(img, (lx, ly), 1, (255, 0, 255), -1)

    def draw_face_landmarks(self, image):

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True

        mp_face_mesh = self.mp_face_mesh
        mp_drawing_styles = self.mp_drawing_styles
        results = self.results['results']
        if results and results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
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

    def process(self):
        cap = self.cv
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
                self.image = image

                temp_results = face_mesh.process(self.image)
                if temp_results:
                    face_landmarks = temp_results.multi_face_landmarks[0]
                else:
                    face_landmarks = None

                if face_landmarks:

                    if self.draw_video_flag:
                        if self.draw_mini_face_flag:
                            self.draw_mini_face(face_landmarks, self.image)

                        if self.draw_landmarks_flag:
                            self.draw_face_landmarks(self.image)

                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                self.results += [{'id': frame_ind, 'image':self.image, 'results': temp_results }]

                cv2.imshow('MediaPipe Face Mesh', self.image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                frame_ind += 1
        cap.release()
