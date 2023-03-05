# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requi_RED by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe solution drawing styles."""

from typing import Mapping, Tuple

from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark

_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

# FaceMesh connections
_THICKNESS_TESSELATION = 1
_THICKNESS_CONTOURS = 2
_FACEMESH_CONTOURS_CONNECTION_STYLE = {
    face_mesh_connections.FACEMESH_LIPS:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYE:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_LEFT_EYEBROW:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYE:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    face_mesh_connections.FACEMESH_FACE_OVAL:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS)
}

def get_facemesh_contours_connection_style(color_left=_GREEN, color_right=_RED, color_face=_WHITE):

    FACEMESH_CONTOURS_CONNECTION_STYLE = {
        face_mesh_connections.FACEMESH_LIPS:
            DrawingSpec(color=color_face, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_LEFT_EYE:
            DrawingSpec(color=color_left, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_LEFT_EYEBROW:
            DrawingSpec(color=color_left, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_RIGHT_EYE:
            DrawingSpec(color=color_right, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
            DrawingSpec(color=color_right, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_FACE_OVAL:
            DrawingSpec(color=color_face, thickness=_THICKNESS_CONTOURS)
    }
    return FACEMESH_CONTOURS_CONNECTION_STYLE



# Pose
_THICKNESS_POSE_LANDMARKS = 2
_POSE_LANDMARKS_LEFT = frozenset([
    PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX
])

_POSE_LANDMARKS_RIGHT = frozenset([
    PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_RIGHT, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])


def get_custom_face_mesh_contours_style(FACEMESH_CONTOURS_CONNECTION_STYLE=_FACEMESH_CONTOURS_CONNECTION_STYLE
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default face mesh contours drawing style.

  Returns:
      A mapping from each face mesh contours connection to its default drawing
      spec.
  """
  face_mesh_contours_connection_style = {}
  for k, v in FACEMESH_CONTOURS_CONNECTION_STYLE.items():
    for connection in k:
      face_mesh_contours_connection_style[connection] = v
  return face_mesh_contours_connection_style


def get_default_face_mesh_tesselation_style() -> DrawingSpec:
  """Returns the default face mesh tesselation drawing style.

  Returns:
      A DrawingSpec.
  """
  return DrawingSpec(color=_GRAY, thickness=_THICKNESS_TESSELATION)


def get_default_face_mesh_iris_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default face mesh iris connections drawing style.

  Returns:
       A mapping from each iris connection to its default drawing spec.
  """
  face_mesh_iris_connections_style = {}
  left_spec = DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS)
  for connection in face_mesh_connections.FACEMESH_LEFT_IRIS:
    face_mesh_iris_connections_style[connection] = left_spec
  right_spec = DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS)
  for connection in face_mesh_connections.FACEMESH_RIGHT_IRIS:
    face_mesh_iris_connections_style[connection] = right_spec
  return face_mesh_iris_connections_style


def get_default_pose_landmarks_style_costum(col_left=(0, 138, 255), col_right=(231, 217, 0)) -> Mapping[int, DrawingSpec]:
  """Returns the default pose landmarks drawing style.

  Returns:
      A mapping from each pose landmark to its default drawing spec.
  """
  pose_landmark_style = {}
  left_spec = DrawingSpec(
      color=col_left, thickness=_THICKNESS_POSE_LANDMARKS)
  right_spec = DrawingSpec(
      color=col_right, thickness=_THICKNESS_POSE_LANDMARKS)
  for landmark in _POSE_LANDMARKS_LEFT:
    pose_landmark_style[landmark] = left_spec
  for landmark in _POSE_LANDMARKS_RIGHT:
    pose_landmark_style[landmark] = right_spec
  pose_landmark_style[PoseLandmark.NOSE] = DrawingSpec(
      color=_WHITE, thickness=_THICKNESS_POSE_LANDMARKS)
  return pose_landmark_style
