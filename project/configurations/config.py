import os.path as osp

root_folder = 'C:\\Users\\user\\EyeGuide\\samples'
sample_video_name = 'vid.avi'
full_path: str = osp.join(root_folder, sample_video_name)

NUM_LANDMARKS = 478

# pose data extraction from videos
base_config = dict(
    config_video=dict(
        img_width=640,
        img_height=480,
        output_width=640,
        output_height=480,
        video_path=full_path
    ),
    config_display=dict(
        draw_landmarks_flag=True,
        draw_mini_face_flag=True,
        draw_video_flag=True)
)

train_config = dict(
    config_video=dict(
        img_width=640,
        img_height=480,
        output_width=640,
        output_height=480,
        video_path=full_path
    ),
    config_display=dict(
        draw_landmarks_flag=False,
        draw_mini_face_flag=False,
        draw_video_flag=False)
)

path_embed_config = dict(
    n_frames=32 * 2,
    n_landmarks=NUM_LANDMARKS,
    dims=2,
    filter_widths=[3, 3, 3],
    dropout=0.25,
    channels=64,
    out_dim=32,
)
