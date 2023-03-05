gestures_dict = {
    'b': 'blink',
    's': 'smile',
    'f': 'frown',
    'k': 'kiss',
}

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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
left_eye_keypoints = {k: v for k, v in important_keypoints.items() if 'lift_' in v}

inv_eye_keypoints = {v: k for k, v in important_keypoints.items()}

test_files_dict = {'category1': [114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515,
                                 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548],
                   'category2': [203, 208, 211, 212, 213, 214, 218, 224, 403, 404, 405, 406, 407, 408, 409, 412, 550,
                                 551, 553],
                   'category3': [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562]}

all_test_files = [item for sub in [v for v in test_files_dict.values()] for item in sub]
