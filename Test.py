import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch.nn.functional as F
import numpy as np
from DataLoader import *
from Metrics import *
from model.MFBF_Net import *
from sklearn.model_selection import train_test_split

model = MFBF_Net().cuda()
model.eval()

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred, decoder = self.model(image)
        loss = DiceLoss()(y_pred, y_true)
        print(loss.cpu().numpy(), end = ' ')
        # loss_test.append(loss.item())
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics


trainer = pl.Trainer()


DATA_PATH = ''
data = np.load("DATA_PATH")
images = data["images"]
masks  = data["masks"]

x_train, x_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.2, random_state=46, shuffle=True
)
test_dataset = BUSILoader(x_test, y_test, typeData='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)


CHECKPOINT_PATH = ''
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)
