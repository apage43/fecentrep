import enum
import math
import pickle
from dataclasses import dataclass

import einops
import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from covweighting import CoVWeightingLoss


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_TokenInitialization":
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x: torch.Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class NumericalFeatureTokenizer(nn.Module):
    """Transforms continuous features to tokens (embeddings).
    See `FeatureTokenizer` for the illustration.
    For one feature, the transformation consists of two steps:
    * the feature is multiplied by a trainable vector
    * another trainable vector is added
    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class BatchSwapNoise(nn.Module):
    def __init__(self, train_only=False):
        super().__init__()
        self.train_only = train_only

    def forward(self, x, p):
        if self.training or not self.train_only:
            if isinstance(x, dict):
                out = {c: self.forward(x[c], p) for c in x.keys()}
                out["p"] = p
                return out
            corrupt = torch.bernoulli(
                torch.ones((x.shape)).to(x.device)
                * (p[:, None] if isinstance(p, torch.Tensor) else p)
            )
            noised = torch.where(corrupt == 1, x[torch.randperm(x.shape[0])], x)
            return noised
        else:
            if isinstance(x, dict):
                x["p"] = p
            return x


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


@dataclass
class FECNetConfig:
    n_entities: int
    n_etypes: int
    n_txtypes: int
    n_conts: int
    n_bins: int
    emb_dim: int
    predict_mask: bool = True
    transformer_dropout: float = 0.1
    transformer_attn_heads: int = 8
    bb_layers: int = 6


class Residual(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x) + x


class FECTokenEncoder(nn.Module):
    "Converts input to a seqeuence of ent_dim-dimensional vectors."

    def __init__(self, config: FECNetConfig):
        super().__init__()
        self.config = config
        self.entity_embedding = nn.Embedding(config.n_entities, config.emb_dim)
        nn.init.normal_(self.entity_embedding.weight, std=1e-5)
        self.etype_emb = nn.Embedding(config.n_etypes, config.emb_dim)
        nn.init.normal_(self.etype_emb.weight, std=1e-5)
        self.txtype_emb = nn.Embedding(config.n_txtypes, config.emb_dim)
        nn.init.normal_(self.txtype_emb.weight, std=1e-5)
        self.nftokenizer = NumericalFeatureTokenizer(
            config.n_conts,
            config.emb_dim,
            True,
            "uniform",
        )
        self.binemb = nn.Embedding(2, config.emb_dim)

    def forward(self, batch):
        cats = batch["cats"]
        ents = batch["ents"]
        bins = batch["bins"]
        conts = batch["conts"]
        # entity embedding
        ent_emb = self.entity_embedding(ents)
        # entity type embedding
        etype_emb = self.etype_emb(cats[:, 0])
        # transaction type embedding
        txtype_emb = self.txtype_emb(cats[:, 1])
        # numerical features
        num_emb = self.nftokenizer(conts)
        # binary features
        bin_emb = self.binemb(bins.long())
        srce = ent_emb[:, 0]
        dste = ent_emb[:, 1]
        seq = torch.stack([srce, dste, etype_emb, txtype_emb], dim=1)
        seq = torch.cat([seq, bin_emb, num_emb], dim=1)
        return seq, ent_emb[:, 0], ent_emb[:, 1]


class FECTokenDecoder(nn.Module):
    def __init__(
        self,
        config: FECNetConfig,
    ):
        super().__init__()
        self.config = config
        self.predict_mask = config.predict_mask
        edmul = 1
        self.entdec_head = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * edmul),
            nn.Mish(),
            nn.LayerNorm(config.emb_dim * edmul),
            nn.Linear(config.emb_dim * edmul, config.n_entities + 1),
        )
        self.txtype_head = nn.Sequential(
            nn.Linear(config.emb_dim, config.n_txtypes + 1),
        )
        self.etype_head = nn.Sequential(
            nn.Linear(config.emb_dim, config.n_etypes + 1),
        )
        self.contdec_head = nn.Sequential(
            Rearrange("s b d -> b (s d)"),
            nn.Linear(config.emb_dim * config.n_conts, config.n_conts),
        )
        self.bindec_head = nn.Sequential(
            Rearrange("s b d -> b (s d)"),
            nn.Linear(config.emb_dim * config.n_bins, config.n_bins),
        )
        if self.predict_mask:
            mdec_dim = 1
            self.maskdec_head = nn.Sequential(
                nn.Linear(config.emb_dim, mdec_dim),  # same transform over all tokens
                nn.Mish(),
                Rearrange("s b d -> b (s d)"),
                nn.Linear(
                    mdec_dim * (config.n_conts + config.n_bins + 4),
                    config.n_conts + config.n_bins + 4,
                ),  # pre-embed mask - src/dec ents, etype, txtype
            )

    def forward(self, seqz):
        seqz = einops.rearrange(seqz, "b s d -> s b d")
        zsrc = seqz[0]
        zdst = seqz[1]
        src = self.entdec_head(zsrc)
        dst = self.entdec_head(zdst)
        et = self.etype_head(seqz[2])
        tt = self.txtype_head(seqz[3])
        bins = self.bindec_head(seqz[4 : 4 + self.config.n_bins])
        conts = self.contdec_head(
            seqz[4 + self.config.n_bins : 4 + self.config.n_bins + self.config.n_conts]
        )
        if self.maskdec_head:
            mask = self.maskdec_head(seqz)
        else:
            mask = None
        return src, dst, et, tt, bins, conts, mask, zsrc, zdst


class FECNet(nn.Module):
    def __init__(
        self,
        config: FECNetConfig,
    ):
        super().__init__()
        self.activation = nn.Mish
        self.encoder = FECTokenEncoder(config)
        tfenc = ContinuousTransformerWrapper(
            max_seq_len=(config.n_conts + config.n_bins + 4),
            emb_dropout=config.transformer_dropout,
            attn_layers=Encoder(
                rotary_pos_emb=True,
                dim=config.emb_dim,
                depth=config.bb_layers,
                ff_glu=True,
                deepnorm=True,
                ff_mult=4,
                ff_swish=True,
                ff_dropout=config.transformer_dropout,
                attn_dropout=config.transformer_dropout,
                heads=config.transformer_attn_heads,
            ),
            post_emb_norm=True,
        )
        self.backbone = nn.Sequential(
            tfenc,
        )
        self.decoder = FECTokenDecoder(config)

    def forward(self, batch):
        e, srce, dste = self.encoder(batch)
        z = self.backbone(e)
        return (self.decoder(z), srce, dste)

    def all_losses(
        self,
        x,
        srce,
        dste,
        src,
        dst,
        et,
        tt,
        bins,
        conts,
        mask,
        y,
        batchmask,
        zsrc,
        zdst,
        correct_entf,
    ):
        entloss = F.cross_entropy(src, y["ents"][:, 0]) + F.cross_entropy(
            dst, y["ents"][:, 1]
        )
        catloss = F.cross_entropy(et, y["cats"][:, 0]) + F.cross_entropy(
            tt, y["cats"][:, 1]
        )
        contloss = F.mse_loss(conts, y["conts"])
        binloss = F.binary_cross_entropy_with_logits(
            bins, y["bins"]
        ) + F.binary_cross_entropy_with_logits(mask, batchmask)
        return [entloss, catloss, contloss, binloss]


class FECNetPL(pl.LightningModule):
    def __init__(
        self,
        config: FECNetConfig,
        max_epochs: int,
        swapnoise_ratio=0.15,
        lr=1e-3,
        tdl=None,
        vdl=None,
        usesched=None,
        warmup_steps=1000,
    ):
        super().__init__()
        self.save_hyperparameters(
            "config", "max_epochs", "lr", "usesched", "warmup_steps"
        )
        self.warmup_steps = warmup_steps if usesched != "oneycle" else None
        self.lr = lr
        self.max_epochs = max_epochs
        self.usesched = usesched
        self.swapnoise_ratio = swapnoise_ratio
        self.core = FECNet(config)
        self.core = torch.compile(self.core)  # , options={"triton.cudagraphs": True})
        self.bsn = BatchSwapNoise()
        self.tdl = tdl
        self.vdl = vdl
        nloss = 4
        self.cov_weighting_loss = CoVWeightingLoss(num_losses=nloss)

    def train_dataloader(self):
        return self.tdl

    def val_dataloader(self):
        return self.vdl

    def training_step(self, batch, batchidx):
        x = self.bsn(batch, self.swapnoise_ratio)
        y = batch
        mask_ents = (x["ents"] != batch["ents"]).float()
        mask_cats = (x["cats"] != batch["cats"]).float()
        mask_conts = (x["conts"] != batch["conts"]).float()
        mask_bins = (x["bins"] != batch["bins"]).float()
        batchmask = torch.cat((mask_ents, mask_cats, mask_conts, mask_bins), dim=1)
        y = batch
        ((src, dst, et, tt, bins, conts, mask, zsrc, zdst), srce, dste) = self.core(x)
        correct_entf = self.core.encoder.entity_embedding(y["ents"])
        losses = self.core.all_losses(
            x,
            srce,
            dste,
            src,
            dst,
            et,
            tt,
            bins,
            conts,
            mask,
            y,
            batchmask,
            zsrc,
            zdst,
            correct_entf,
        )
        # entloss, catloss, contloss, binloss, embloss
        self.log("entity_loss", losses[0], on_step=True)
        self.log("cat_loss", losses[1], on_step=True)
        self.log("cont_loss", losses[2], on_step=True)
        self.log("bin_loss", losses[3], on_step=True)
        loss = self.cov_weighting_loss(losses)

        self.log("total_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batchidx):
        x = self.bsn(batch, self.swapnoise_ratio)
        y = batch
        mask_ents = (x["ents"] != batch["ents"]).float()
        mask_cats = (x["cats"] != batch["cats"]).float()
        mask_conts = (x["conts"] != batch["conts"]).float()
        mask_bins = (x["bins"] != batch["bins"]).float()
        batchmask = torch.cat((mask_ents, mask_cats, mask_conts, mask_bins), dim=1)
        y = batch
        ((src, dst, et, tt, bins, conts, mask, zsrc, zdst), srce, dste) = self.core(x)
        correct_entf = self.core.encoder.entity_embedding(y["ents"])
        losses = self.core.all_losses(
            x,
            srce,
            dste,
            src,
            dst,
            et,
            tt,
            bins,
            conts,
            mask,
            y,
            batchmask,
            zsrc,
            zdst,
            correct_entf,
        )
        self.log("val_entity_loss", losses[0], on_epoch=True, sync_dist=True)
        self.log("val_cat_loss", losses[1], on_epoch=True, sync_dist=True)
        self.log("val_cont_loss", losses[2], on_epoch=True, sync_dist=True)
        self.log("val_bin_loss", losses[3], on_epoch=True, sync_dist=True)

        weighted_losses = [
            self.cov_weighting_loss.alphas[i] * losses[i] for i in range(len(losses))
        ]
        loss = sum(weighted_losses)
        self.log("total_val_loss", loss, on_epoch=True, sync_dist=True)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        if self.warmup_steps:
            # manually warm up lr without a scheduler
            if self.trainer.global_step < self.warmup_steps:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / float(self.warmup_steps)
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.trainer.model.parameters()),
            eps=1e-5,
            lr=self.lr,
        )
        train_loader_len = len(self.train_dataloader())

        scheduler = None
        if self.usesched == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.lr,
                steps_per_epoch=train_loader_len,
                epochs=self.max_epochs,
            )
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(scheduler=scheduler, interval="step"),
            )
        if self.usesched == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=math.sqrt(0.1), patience=5, verbose=True
            )
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=plateau, monitor="val_entity_loss", interval="epoch"
                ),
            )

        return optimizer


def main():
    lr = 1e-3
    batch_size = 1000
    max_epochs = 40
    jobname = f"tblenc-xt-{max_epochs}ep-plat"
    fecdf = pd.read_parquet("./fecpreprocd.parquet")
    with open("./meta.pkl", "rb") as rf:
        meta = pickle.load(rf)
    fnds = FECNetDataset(meta, fecdf)
    splitgen = torch.Generator().manual_seed(41)
    train_set, val_set = random_split(fnds, [0.9, 0.1], generator=splitgen)
    tdl = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    vdl = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    config = FECNetConfig(
        n_entities=len(meta["ents"]),
        n_etypes=len(meta["etype"]),
        n_txtypes=len(meta["txtype"]),
        n_conts=len(fnds.contcols),
        n_bins=len(fnds.bincols),
        emb_dim=512,
        transformer_dropout=0.25,
        transformer_attn_heads=8,
        bb_layers=10,
    )
    print(config)
    plmodel = FECNetPL(
        config=config,
        max_epochs=max_epochs,
        tdl=tdl,
        vdl=vdl,
        usesched="plateau",
        lr=lr,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_entity_loss",
        save_top_k=1,
        save_last=True,
        dirpath=f"./fec-ckpt/{jobname}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_entity_loss", patience=10, verbose=True, mode="min"
    )
    logger = TensorBoardLogger("tb_logs", name=jobname)
    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        # strategy="deepspeed_stage_2_offload",
        precision="16-mixed",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(plmodel, tdl)


if __name__ == "__main__":
    main()
