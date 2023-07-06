from project.utils.read_gestures_class import CustomGesture, FacialWriting, ReadGestures


# Set commands manually for facial commands
# You can set any command using the basic gesture components, for more, read pars_command_array doc string

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
# facial_writing = FacialWriting()
# facial_writing.run()

read_gesture = ReadGestures(custom_gesture_array)
read_gesture.run()
