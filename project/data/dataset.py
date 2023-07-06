import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os.path as osp
import json
import numpy as np
import os
import cv2
import mediapipe as mp
from tqdm import tqdm
from utils.visualize_reconstruction import visualize_pose_sequence
from utils.constants import TRAINED_LANDMARKS
from torch.utils.data import random_split, Subset


class HorizontalMirror:
    def __init__(self, width=1200):
        self.name = 'horizontal_mirror'
        self.width = width
        self.prob = 0.5

    def __call__(self, x):
        if torch.rand(1) > self.prob:
            x[:, :, 0] = self.width - x[:, :, 0]
        return x


class Move:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height

    def __call__(self, x):
        max_x = np.max(x[:, :, 0])
        max_y = np.max(x[:, :, 1])

        min_x = np.min(x[:, :, 0])
        min_y = np.min(x[:, :, 1])

        x_max_diff = (self.width - max_x) * 0.5
        x_min_diff = (0 - min_x) * 0.5

        y_max_diff = (self.height - max_y) * 0.5
        y_min_diff = (0 - min_y) * 0.5

        x_diff = (x_max_diff - x_min_diff) * np.random.rand(1) + x_min_diff
        y_diff = (y_max_diff - y_min_diff) * np.random.rand(1) - y_min_diff

        x[:, :, 0] = x[:, :, 0] + x_diff
        x[:, :, 1] = x[:, :, 1] + y_diff
        return x


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
    def __init__(self, data_path, n_frames=64, n_landmarks=478, n_dims=2, train=False, transforms=None, limit=None, data_in=None):
        self.n_frames = n_frames
        self.n_landmarks = n_landmarks
        self.n_dims = n_dims

        self.data_path = data_path
        self.train = train
        self.transforms = transforms if transforms is not None else []

        self.all_videos_paths = glob.glob(osp.join(self.data_path, '*', 'vid.json'))
        self.limit = limit
        if data_in is None:
            self._load_dataset()
        else:
            self.data = data_in

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        for trans in self.transforms:
            x = trans(x)
        return x

    def _load_dataset(self):
        all_poses = []
        bad_files = []
        # al_time_stamps = []
        if self.limit is None:
            limit = -1
            n_files_to_load = len(self.all_videos_paths)
        else:
            limit = self.limit
            n_files_to_load = limit
        n_files = 0

        filename = f'n_files{n_files_to_load}.pickle'

        cache_folder_path = 'C:\\Users\\user\\EyeGuide\\cache'

        filename = osp.join(cache_folder_path, filename)

        if osp.exists(filename):
            print('Loading from cache')
            with open(filename, 'rb') as f:
                all_poses = pickle.load(f)

        else:
            print(f'Could not find cache file {filename}')
            with tqdm(total=n_files_to_load) as pbar:
                for data_path in self.all_videos_paths:
                    try:
                        with open(data_path, 'r') as handle:
                            loaded_results = json.load(handle)
                            pose = [p['results'] for p in loaded_results]
                            pose = np.array(pose)
                            tot_n_frames, n_landmarks, n_dims = pose.shape
                            assert n_landmarks == self.n_landmarks and n_dims == self.n_dims, 'Invalid pose shape'

                            pose = pad_pose(pose, self.n_frames, 'symmetric')
                            pose = pose.reshape(-1, self.n_frames, n_landmarks, n_dims)
                            all_poses.append(pose)
                            n_files += 1
                    except:
                        bad_files.append(data_path)
                    pbar.update()
                    if 0 < limit <= n_files:
                        break

            if len(bad_files):
                print(f'something was wrong with {len(bad_files)} files')
                print(bad_files)

            with open(filename, 'wb') as f:
                pickle.dump(all_poses, f)

        # load subset of landmarks including only eyes, lips etc
        all_poses = np.vstack(all_poses)
        all_poses = all_poses[:, :, TRAINED_LANDMARKS, :]
        self.data = all_poses


def get_dataloader_split_and_save():
    ds, dl = get_dataloader(data_path=None, n_landmarks=478, sample_size_limit=None, batch_size=16,
                            n_frames=128)
    train_size = int(len(ds) * 0.8)
    valid_size = len(ds) - train_size

    train_ds, valid_ds = random_split(ds, [train_size, valid_size])
    batch_size = 16
    # train_dl, valid_dl = DataLoader(train_ds, batch_size=batch_size), DataLoader(valid_ds, batch_size=batch_size)
    save_path = 'C:\\Users\\user\\EyeGuide\\assets\\train_validate_indecies.pickle'

    data_train = train_ds.indices
    data_val = valid_ds.indices
    inds = {'train': data_train, 'val': data_val}

    with open(save_path, 'wb') as f:
        pickle.dump(inds, f)

    # return train_dl, valid_dl, train_ds, valid_ds
    return train_ds, valid_ds


def get_dataloader(data_path=None, n_landmarks=478, sample_size_limit=None, batch_size=None, n_frames=None):
    if data_path is None:
        data_path = 'C:\\Users\\user\\EyeGuide\\assets\\300VW_Dataset_2015_12_14'
    if batch_size is None:
        batch_size = 16
    if n_frames is None:
        n_frames = 128
    num_workers = 8

    ds = FaceDataset(data_path, n_frames=n_frames, limit=sample_size_limit, n_landmarks=n_landmarks,
                     transforms=[HorizontalMirror()])
    ds = FaceDataset(data_path, n_frames=n_frames, limit=sample_size_limit, n_landmarks=n_landmarks,
                     )
    dl = DataLoader(ds, batch_size=batch_size)
    return ds, dl

def get_train_val_files(batch_size=16):
    with open('C:\\Users\\user\\EyeGuide\\assets\\train_validate_indecies.pickle','rb') as f:
        inds = pickle.load(f)
    ds, dl = get_dataloader(data_path=None, n_landmarks=478, sample_size_limit=None, batch_size=16,
                            n_frames=128)
    train_ds, val_ds = Subset(ds, inds['train']), Subset(ds, inds['val'])

    train_dl, valid_dl = DataLoader(train_ds, batch_size=batch_size), DataLoader(val_ds, batch_size=batch_size)
    return train_dl, valid_dl, train_ds, val_ds


if __name__ == '__main__':
    pass
    # data_path = 'C:\\Users\\user\\EyeGuide\\assets\\300VW_Dataset_2015_12_14'
    # sample_size = 5
    # ds = FaceDataset(data_path, n_frames=32, limit=sample_size)
    # dl = DataLoader(ds, batch_size=sample_size)
    # data = next(iter(dl))
    # visualize_pose_sequence(data[:, :, :, :], inflate_ratio=5, center=True, loops=10)
    # train_ds, train_dl = get_dataloader(data_path=None, n_landmarks=478, sample_size_limit=None, batch_size=16, n_frames=128)
    get_dataloader_split_and_save()
#