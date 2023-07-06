import torch.cuda
import lightning.pytorch as pl
from finetuning_scheduler import FinetuningScheduler
from models.mae_model import mae_vit_tdcnn
import torch.optim as optim
from data.dataset import get_dataloader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import os.path as osp
from utils.constants import TRAINED_LANDMARKS
import pickle
from data.dataset import FaceDataset, get_train_val_files


epochs = 1000
lr = 0.01

model = mae_vit_tdcnn()
model = model.double()
opt = optim.Adam(model.parameters(), lr=lr)
n_landmarks = len(TRAINED_LANDMARKS)
# train_ds, train_dl = get_dataloader(data_path=None, n_landmarks=478,
#                                     sample_size_limit=None, batch_size=16, n_frames=32*4)
train_dl, valid_dl, train_ds, valid_ds = get_train_val_files()

# datasets_path= 'C:\\Users\\user\\EyeGuide\\assets\\train_validate_datasets'
# with open(datasets_path, 'rb') as f:
#     train_dl, valid_dl, train_ds, valid_ds = pickle.load(f)
device = 0
display_rate = 10
iteration = 0

# trainer = pl.Trainer(min_epochs=1, max_epochs=50)
checkpoint_callback = ModelCheckpoint(dirpath=osp.join('..','checkpoints'), save_top_k=2, monitor="val_loss")
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping], min_epochs=1)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)



