import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class UNetLightning(pl.LightningModule):
    def __init__(self, in_channels, n_classes, layers, lr=1e-3):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()  # Automatically saves init args
        self.model = UNet(in_c=in_channels, n_classes=n_classes, layers=layers)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch  # Assuming batch is (input, target)
        logits = self(images) #Calls self.forward internally
        loss = F.cross_entropy(logits, y)  # Adjust loss depending on your task
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
