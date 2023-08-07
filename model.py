import os

import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision import transforms
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1, padding=1, downsample = None):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                                    nn.BatchNorm2d(out_channels),
                                    # nn.ReLU(inplace=False)
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                                    nn.BatchNorm2d(out_channels))

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        if self.downsample:
            residual = self.downsample(x)
        out+=residual
        out = self.relu(out)
        return out

class LightningDavidNet(LightningModule):

    def __init__(self,data_dir=PATH_DATASETS, hidden_size=16, learning_rate=2e-4,kernel_size=3, stride=1, padding=1, downsample = None):
        super().__init__()
        self.learning_rate =learning_rate
        self.data_dir = data_dir
        self.hidden_size = hidden_size

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.prep = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=False))
        self.l1X = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                                nn.MaxPool2d(kernel_size = 2),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=False))
        self.r1 = ResBlock(128, 128,kernel_size=3, stride=1, padding=1, downsample = None)
        self.l2X = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
                                nn.MaxPool2d(kernel_size = 2),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=False))
        self.l3X = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
                                nn.MaxPool2d(kernel_size = 2),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=False))
        self.r2 = ResBlock(512, 512,kernel_size=3, stride=1, padding=1, downsample = None)
        self.maxPool = nn.MaxPool2d(kernel_size = 4)
        self.fc1 = nn.Linear(512,10)

        self.accuracy = Accuracy(task = "multiclass",num_classes = self.num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.l1X(x)
        residual = x
        x = self.r1(x)
        x= residual+ x
        x = self.l2X(x)
        x = self.l3X(x)
        residual = x
        x = self.r2(x)
        x=residual+x
        x = self.maxPool(x)
        # # x = self.avgpool(x)
        x = x.view(-1,512)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.03, weight_decay=1e-4)
        steps_per_epoch = len(train_loader)
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        # return optimizer

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits,dim = 1)
        self.accuracy(preds,y)
        self.log("val_loss",loss, prog_bar = True)
        self.log("val_arr",self.accuracy,prog_bar = True)

    def test_step(self,batch,batch_idx):
        return self.validation_step(batch,batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x,y = batch
        output = self(x)
        return x,y,output.argmax(dim=1),output