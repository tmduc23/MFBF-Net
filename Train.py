import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from DataLoader import *
from Metrics import *
from model.MFBF_Net import *
import numpy as np
from sklearn.model_selection import train_test_split


import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        # loss = DiceLoss()(y_pred, y_true)
        # loss = bce_tversky_loss(y_pred, y_true)
        loss = dice_tversky_loss(y_pred, y_true)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        return loss, dice, iou

    def training_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice, "train_iou": iou}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"val_loss":loss, "val_dice": dice, "val_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss":loss, "test_dice": dice, "test_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(1e-3))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor = 0.5, patience=8, verbose =True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers

model = MFBF_Net().cuda()

DATA_PATH = ''


data = np.load("DATA_PATH")
images = data["images"]
masks  = data["masks"]

x_train, x_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.2, random_state=46, shuffle=True
)

# Dataset & Data Loader
train_dataset = BUSILoader(x_train, y_train, typeData='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True)

val_dataset = BUSILoader(x_test, y_test, typeData='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)


trainer_path = "/content/weights"
os.makedirs(trainer_path, exist_ok = True)
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint(trainer_path, filename="ckpt{val_dice:0.4f}_wo_all",
                                                            monitor="val_dice", mode = "max", save_top_k =1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False)


progress_bar = pl.callbacks.TQDMProgressBar()
PARAMS = {"benchmark": True, "enable_progress_bar" : True,"logger":True,
          "callbacks" : [check_point, progress_bar],
          "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":200,
          "precision":16,
          }
trainer = pl.Trainer(**PARAMS)
segmentor = Segmentor(model=model)

trainer.fit(segmentor, train_loader, val_loader)