from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os.path as osp
import os
import cv2
import mediapipe as mp

class GestureDataset(Dataset):

    def __init__(self, data_path,config):
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

class FaceDataset(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        if 'train' in config:
            self.train = config['train']
        else:
            self.train = False

        self.all_videos_paths = glob.glob(osp.join(self.data_path,'*','vid*'))
        self.mediapipe_object

    def __len__(self):
        return len(self.all_videos_paths)

    def __getitem__(self, idx):
        vid_path = self.all_videos_path[idx]


        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

