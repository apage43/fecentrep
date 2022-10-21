import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
class CoVWeightingLoss(nn.Module):

    """
        Adapted from https://github.com/rickgroen/cov-weighting/blob/main/losses/covweighting_loss.py
        
        Weighs the losses to the Cov-Weighting method, where the statistics are maintained through Welford's 
        algorithm. But now for 32 losses.
    """

    def __init__(self, num_losses, mean_decay=None):
        super().__init__()

        self.num_losses = num_losses
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = mean_decay

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
        self.running_std_l = None

    def forward(self, unweighted_losses):
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay is not None:
            mean_param = self.mean_decay
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return loss

class FECNetDataset(torch.utils.data.Dataset):
    def __init__(self, lbls, df):
        super().__init__()

        catcols = lbls["cols"]["cats"]
        entcols = lbls["cols"]["ents"]
        bincols = lbls["cols"]["bins"]
        contcols = lbls["cols"]["conts"]
        print("Column types:")
        print(lbls["cols"])
        self.catcols = catcols
        self.entcols = entcols
        self.bincols = bincols
        self.contcols = contcols
        self.lbls = lbls
        self.df = df
        self.cats = torch.LongTensor(self.df[catcols].values)
        self.ents = torch.LongTensor(self.df[entcols].values)
        self.bins = torch.FloatTensor(self.df[bincols].values)
        self.conts = torch.FloatTensor(self.df[contcols].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return dict(
            cats=self.cats[idx],
            ents=self.ents[idx],
            bins=self.bins[idx],
            conts=self.conts[idx],
        )


class BatchSwapNoise(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            if isinstance(x, dict):
                return {c: self.forward(x[c]) for c in x.keys()}
            mask = torch.rand(x.size()) > (1 - self.p)
            idx = torch.add(
                torch.arange(x.nelement()),
                (
                    torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor)
                    * (mask.type(torch.LongTensor) * x.size(1))
                ).view(-1),
            )
            idx[idx >= x.nelement()] = idx[idx >= x.nelement()] - x.nelement()
            return x.view(-1)[idx].view(x.size())
        else:
            return x

class FECDenoisingAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        n_entities,
        n_etypes,
        n_txtypes,
        n_conts,
        n_bins,
        max_epochs,
        ent_feats=256,
        etype_feats=12,
        txtype_feats=32,
        hdim=1024,
        zdim=2048,
        swapnoise_ratio=0.15,
        tdl=None,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.entity_embedding = nn.Embedding(n_entities, ent_feats)
        self.etype_emb = nn.Embedding(n_etypes, etype_feats)
        self.txtype_emb = nn.Embedding(n_txtypes, txtype_feats)

        self.srcdec_head = nn.Linear(zdim, n_entities + 1)
        self.dstdec_head = nn.Linear(zdim, n_entities + 1)
        self.etdec_head = nn.Linear(zdim, n_etypes + 1)
        self.ttdec_head = nn.Linear(zdim, n_txtypes + 1)
        self.contdec_head = nn.Linear(zdim, n_conts)
        self.bindec_head = nn.Linear(zdim, n_bins)

        self.encoder = nn.Sequential(
            nn.Linear(
                ent_feats * 2 + etype_feats + txtype_feats + n_conts + n_bins, hdim
            ),
            nn.BatchNorm1d(hdim),
            nn.Dropout(),  
            nn.Mish(),
            nn.Linear(hdim, hdim),
            nn.BatchNorm1d(hdim),
            nn.Dropout(),
            nn.Mish(),
            nn.Linear(hdim, hdim),
            nn.BatchNorm1d(hdim),
            nn.Dropout(),
            nn.Mish(),
            nn.Linear(hdim, zdim),
        )        
        self.bsn = BatchSwapNoise(swapnoise_ratio)
        self.tdl = tdl
        self.cov_weighting_loss = CoVWeightingLoss(num_losses=4)

    def train_dataloader(self):
        return self.tdl

    def forward(self, x):
        entfeat = self.entity_embedding(x["ents"])
        entypef = self.etype_emb(x["cats"][:, 0])
        txtypef = self.txtype_emb(x["cats"][:, 1])

        i = torch.cat(
            (entfeat.flatten(1), entypef, txtypef, x["conts"], x["bins"]), dim=1
        )
        h = self.encoder(i)

        src = self.srcdec_head(h)
        dst = self.dstdec_head(h)
        et = self.etdec_head(h)
        tt = self.ttdec_head(h)
        bins = self.bindec_head(h)
        conts = self.contdec_head(h)
        return src, dst, et, tt, bins, conts

    def all_losses(self, src, dst, et, tt, bins, conts, y):
        entloss = F.cross_entropy(src, y["ents"][:, 0])
        entloss += F.cross_entropy(dst, y["ents"][:, 1])
        catloss = F.cross_entropy(et, y["cats"][:, 0])
        catloss += F.cross_entropy(tt, y["cats"][:, 1])
        contloss = F.mse_loss(conts, y["conts"])
        binloss = F.binary_cross_entropy_with_logits(bins, y["bins"])

        self.log("entity_loss", entloss, on_step=True)
        self.log("cat_loss", catloss, on_step=True)
        self.log("cont_loss", contloss, on_step=True)
        self.log("bin_loss", binloss, on_step=True)
        return [entloss, catloss, contloss, binloss]
        
    def training_step(self, batch, batchidx):
        x = self.bsn(batch)
        y = batch
        src, dst, et, tt, bins, conts = self.forward(x)
        losses = self.all_losses(src, dst, et, tt, bins, conts, y)
        loss = self.cov_weighting_loss(losses)
        self.log("total_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        train_loader_len = len(self.train_dataloader())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                1e-3,
                steps_per_epoch=train_loader_len,
                epochs=self.max_epochs,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "scheduler": dict(scheduler=scheduler, interval="step"),
            }
        else:
            return optimizer




def run(packed_embedding_init=True, jobname="fecencoder-mtl1"):
    epochs = 150
    fecdf = pd.read_parquet("./fecpreprocd.parquet")

    with open("./meta.pkl", "rb") as rf:
        meta = pickle.load(rf)
    fnds = FECNetDataset(meta, fecdf)
    tdl = DataLoader(fnds, batch_size=20000, shuffle=True, num_workers=8)
    logger = TensorBoardLogger("tb_logs", name=jobname)
    checkpoint_callback = ModelCheckpoint(dirpath=f"./fec-ckpt/{jobname}")

    model = FECDenoisingAutoEncoder(
        len(meta["ents"]),
        len(meta["etype"]),
        len(meta["txtype"]),
        len(fnds.contcols),
        len(fnds.bincols),
        max_epochs = epochs,
        tdl=tdl
    )

    if packed_embedding_init:
        def packinit(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

        model.apply(packinit)

    trainer = pl.Trainer(
        gpus=-1,
        accelerator="dp",  # precision=16,
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=20,
    )

    trainer.fit(model, tdl)
    logger.experiment.add_embedding(model.entity_embedding.weight, meta["ents"])


if __name__ == '__main__':
    run()