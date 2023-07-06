gestures_dict = {
    'b': 'blink',
    's': 'smile',
    'f': 'frown',
    'k': 'kiss',
}

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# nose
NOSE = [8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 278, 79, 20, 242, 370, 462, 250, 237, 274, 241, 461]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

important_keypoints = {
    468: 'right_eye_center',
    469: 'right_eye_left',
    470: 'right_eye_top',
    471: 'right_eye_right',
    472: 'right_eye_bottom',

    473: 'left_eye_center',
    474: 'left_eye_left',
    475: 'left_eye_top',
    476: 'left_eye_right',
    477: 'left_eye_bottom',

    9: 'forehead',
    200: 'chin',

}



right_eye_keypoints = {k: v for k, v in important_keypoints.items() if 'right_' in v}
left_eye_keypoints = {k: v for k, v in important_keypoints.items() if 'left_' in v}

inv_eye_keypoints = {v: k for k, v in important_keypoints.items()}

test_files_dict = {'category1': [114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515,
                                 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548],
                   'category2': [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550,
                                 551, 553],
                   'category3': [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562]}

all_test_files = [item for sub in [v for v in test_files_dict.values()] for item in sub]

COLORS = dict(
    RED=(48, 48, 255),
    GREEN=(48, 255, 48),
    BLUE=(192, 101, 21),
    YELLOW=(0, 204, 255),
    GRAY=(128, 128, 128),
    PURPLE=(128, 64, 128),
    PEACH=(180, 229, 255),
    WHITE=(224, 224, 224)
)

MESH_ANNOTATIONS = {
  'silhouette': [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ],

  'lipsUpperOuter': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

  'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
  'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
  'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
  'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
  'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
  'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
  'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],

  'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
  'rightEyebrowLower': [35, 124, 46, 53, 52, 65],

  'rightEyeIris': [473, 474, 475, 476, 477],

  'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
  'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
  'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
  'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
  'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
  'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
  'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],

  'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
  'leftEyebrowLower': [265, 353, 276, 283, 282, 295],

  'leftEyeIris': [468, 469, 470, 471, 472],

  'midwayBetweenEyes': [168],

  'noseTip': [1],
  'noseBottom': [2],
  'noseRightCorner': [98],
  'noseLeftCorner': [327],

  'rightCheek': [205],
  'leftCheek': [425]
}


TRAINED_LANDMARKS = [FACE_OVAL, LIPS, RIGHT_EYE, RIGHT_EYEBROW, LEFT_EYE, LEFT_EYEBROW, important_keypoints.keys()]
# TRAINED_LANDMARKS = list(MESH_ANNOTATIONS.values())
TRAINED_LANDMARKS = list(set([item for sublist in TRAINED_LANDMARKS for item in sublist]))

# print(len(TRAINED_LANDMARKS))