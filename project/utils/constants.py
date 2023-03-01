gestures_dict = {
    'b': 'blink',
    's': 'smile',
    'f': 'frown',
    'k': 'kiss',
}

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

