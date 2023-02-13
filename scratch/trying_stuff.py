from utils.mediapipe_utils import MediapipeVideoExtractor
from configurations.config import base_config
import json
import glob
import os.path as osp
import os

root_folder = 'C:\\Users\\user\\EyeGuide'
# video_path = 'vid.avi'
# full_path = osp.join(root_folder, video_path)

save_folder = osp.join(root_folder, 'project', 'assets')
data_folder = osp.join(root_folder, 'data', '300VW_Dataset_2015_12_14')

all_vid_paths = glob.glob(osp.join(data_folder, '*', 'vid*'))
for path_to_video in all_vid_paths:
    base_config['video_path'] = path_to_video

    save_folder = path_to_video.replace('data', 'assets')
    save_folder = osp.split(save_folder)[0]
    save_name = osp.join(save_folder, 'vid.json')

    if not os.path.exists(save_name):
        mp_obj = MediapipeVideoExtractor(base_config, debug=False)
        results = mp_obj.process()
    else:
        continue

    if not os.path.exists(save_folder):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(save_folder)

    with open(save_name, 'w') as handle:
        json.dump(results, handle)

# with open(save_name, 'r') as handle:
#     loaded_results = json.load(handle)

# print(loaded_results)
