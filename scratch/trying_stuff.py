from utils.mediapipe_utils import MediapipeVideoObj
import os.path as osp

root_folder = 'C:\\Users\\user\\EyeGuide\\data'
video_path = '300VW_Dataset_2015_12_14\\300VW_Dataset_2015_12_14\\001\\vid.avi'
full_path = osp.join(root_folder, video_path)

config = {
    'img_width': 640,
    'img_height': 480,
    'output_width': 640,
    'output_height': 480,
    'video_path': full_path
}


mp_obj = MediapipeVideoObj(config)
mp_obj.process()