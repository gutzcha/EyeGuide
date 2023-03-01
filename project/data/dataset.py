from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os.path as osp
import json
import numpy as np
import os
import cv2
import mediapipe as mp


def pad_pose(pose, n_frames=64, pad_mode='symmetric', constant_values=0):
    tot_n_frames, n_landmarks, n_dims = pose.shape
    if tot_n_frames > n_frames:
        num_pad = tot_n_frames % n_frames
    else:
        num_pad = n_frames % tot_n_frames
    if pad_mode == 'constant':
        return np.pad(pose, pad_width=((0, n_frames - num_pad), (0, 0), (0, 0)), mode=pad_mode,
                      constant_values=constant_values)
    else:
        return np.pad(pose, pad_width=((0, n_frames - num_pad), (0, 0), (0, 0)), mode=pad_mode)


class GestureDataset(Dataset):

    def __init__(self, data_path, config):
        self.data_path = data_path

        if 'n_frames' in config:
            self.n_frames = config['n_frames']
        else:
            self.n_frames = 30
        if 'transform' in config:
            self.transform = config['transform']

        self._load_dataset()

    def _load_dataset(self):

        with open(self.data_path, 'rb') as handle:
            self.data = pickle.load(handle)

    def __len__(self):
        pass  # TODO

    def __getitem__(self, item):
        pass  # TODO


class FaceDataset(Dataset):
    def __init__(self, data_path, n_frames=64, n_landmarks=478, n_dims=2, train=False, transforms=None):
        self.n_frames = n_frames
        self.n_landmarks = n_landmarks
        self.n_dims = n_dims

        self.data_path = data_path
        self.train = train
        self.transforms = transforms

        self.all_videos_paths = glob.glob(osp.join(self.data_path, 'vid.json'))

        self._load_dataset()


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_dataset(self):
        all_poses = []
        # al_time_stamps = []
        for data_path in self.all_videos_paths:
            with open(data_path, 'r') as handle:
                loaded_results = json.load(handle)
                pose = [p['results'] for p in loaded_results]
                pose = np.array(pose)
                tot_n_frames, n_landmarks, n_dims = pose.shape
                assert n_landmarks == self.n_landmarks and n_dims == self.n_dims, 'Invalid pose shape'

                pose = pad_pose(pose, self.n_frames, 'symmetric')
                pose = pose.reshape(-1, self.n_frames, n_landmarks, n_dims)
                all_poses.append(pose)

        self.data = np.vstack(all_poses)


if __name__ == '__main__':
    workspace = 'C:\\workspace\\EyeGuide\\'
    data_path = 'samples'
    ds = FaceDataset(osp.join(workspace, data_path))
    dl = DataLoader(ds)
    next(iter(dl))
