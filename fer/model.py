import math
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, ContinuousTransformerWrapper


@dataclass
class Config:
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 10
    entity_dim: int = 256
    embedding_init_std: Optional[float] = 0.01
    tied_encoder_decoder_emb: bool = False


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = out.sin()
        return out


class DatetimeEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.compdim = config.transformer_dim // 6
        self.yddim = self.compdim + (config.transformer_dim - (self.compdim * 6))

        self.yearenc = Siren(1, self.compdim, w0=10.0)
        self.monthenc = Siren(1, self.compdim, w0=12.0)
        self.weekenc = Siren(1, self.compdim, w0=52.0)
        self.monthdayenc = Siren(1, self.compdim, w0=32.0)
        self.weekdayeenc = Siren(1, self.compdim, w0=7.0)
        self.yeardayeenc = Siren(1, self.yddim, w0=365.0)

    def forward(self, batch: Dict[str, torch.Tensor]):
        yearf = self.yearenc(batch["dt_year"])
        monthf = self.monthenc(batch["dt_year"])
        weekf = self.weekenc(batch["dt_week"])
        mdf = self.monthdayenc(batch["dt_day"])
        wdf = self.weekdayeenc(batch["dt_weekday"])
        ydf = self.yeardayeenc(batch["dt_yearday"])
        return torch.cat([yearf, monthf, weekf, mdf, wdf, ydf], dim=1)


class FECEncoder(nn.Module):
    def __init__(self, config: Config, n_entities: int, n_etype: int, n_ttype: int):
        super().__init__()
        self.entity_embeddings = nn.Embedding(
            n_entities, embedding_dim=config.entity_dim
        )
        if config.embedding_init_std is not None:
            nn.init.normal_(
                self.entity_embeddings.weight, std=config.embedding_init_std
            )
        self.etype_embeddings = nn.Embedding(
            n_etype, embedding_dim=config.transformer_dim
        )
        if config.embedding_init_std is not None:
            nn.init.normal_(self.etype_embeddings.weight, std=config.embedding_init_std)
        self.ttype_embeddings = nn.Embedding(
            n_ttype, embedding_dim=config.transformer_dim
        )
        if config.embedding_init_std is not None:
            nn.init.normal_(self.ttype_embeddings.weight, std=config.embedding_init_std)
        self.posneg_encoding = nn.Embedding(2, config.transformer_dim)
        self.amt_encoder = nn.Linear(1, config.transformer_dim)
        self.dt_encoder = DatetimeEncoder(config)

    def forward(self, batch: Dict[str, torch.Tensor]):
        srcf = self.entity_embeddings(batch["src"]).squeeze()
        dstf = self.entity_embeddings(batch["dst"]).squeeze()
        etypef = self.etype_embeddings(batch["etype"]).squeeze()
        ttypef = self.ttype_embeddings(batch["ttype"]).squeeze()
        amtf = F.gelu(self.amt_encoder(batch["amt"]))
        amtsignf = F.gelu(self.posneg_encoding(batch["amt_pos"]).squeeze())
        dtf = self.dt_encoder(batch)
        seq = torch.stack([srcf, dstf, etypef, ttypef, amtf + amtsignf, dtf])
        return seq


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


def tdtype(k):
    if k in {"src", "dst", "etype", "ttype", "amt_pos"}:
        return torch.long
    return torch.float


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, dd):
        self.dd = dd

    def __len__(self):
        return len(next(iter(self.dd.values())))

    def __getitem__(self, idx):
        batch = {
            k: torch.tensor(v[idx], dtype=tdtype(k)).unsqueeze(-1)
            for k, v in self.dd.items()
        }
        return batch


class FECDecoder(nn.Module):
    def __init__(self, config: Config, n_entities: int, n_etype: int, n_ttype: int):
        super().__init__()
        self.config = config

        def nldec(m):
            return nn.Sequential(
                nn.Linear(config.transformer_dim, config.transformer_dim, bias=False), nn.GELU(), m
            )

        if config.tied_encoder_decoder_emb:
            self.entdec = None
        else:
            self.entdec = nn.Linear(config.transformer_dim, n_entities + 1)
        self.etdec = nn.Linear(config.transformer_dim, n_etype + 1)
        self.ttdec = nn.Linear(config.transformer_dim, n_ttype + 1)
        self.amtdec = nldec(nn.Linear(config.transformer_dim, 1))
        self.amtbindec = nn.Linear(config.transformer_dim, 1)
        self.datetimedec = nldec(nn.Linear(config.transformer_dim, 6)) # y,m,d,md,wd,yd

    def forward(self, x, encoder: FECEncoder):
        if self.entdec is not None:
            srclogits = self.entdec(x[0])
            dstlogits = self.entdec(x[1])
        else:
            srclogits = F.linear(x[0], encoder.entity_embeddings.weight)
            dstlogits = F.linear(x[1], encoder.entity_embeddings.weight)
        etlogits = self.etdec(x[2])
        ttlogits = self.ttdec(x[3])
        amtd = self.amtdec(x[4])
        amtpos = self.amtbindec(x[4])
        dtime = self.datetimedec(x[5])
        return srclogits, dstlogits, etlogits, ttlogits, amtd, amtpos, dtime


class TabularDenoiser(nn.Module):
    def __init__(self, config: Config, n_entities: int, n_etype: int, n_ttype: int):
        super().__init__()
        self.config = config
        self.encoder = FECEncoder(
            config, n_entities=n_entities, n_etype=n_etype, n_ttype=n_ttype
        )
        self.decoder = FECDecoder(
            config, n_entities=n_entities, n_etype=n_etype, n_ttype=n_ttype
        )
        self.bnoise = BatchSwapNoise()

        self.tfenc = ContinuousTransformerWrapper(
            max_seq_len=6,
            emb_dropout=0.2,
            attn_layers=Encoder(
                dim=config.transformer_dim,
                depth=config.transformer_layers,
                use_rmsnorm=True,
                deepnorm=True,
                ff_mult=4,
                attn_flash=True,
                heads=config.transformer_heads,
                ff_dropout=0.2,
                no_bias_ff=True,
                add_zero_kv=True,
                zero_init_branch_output=True,
            ),
            post_emb_norm=False,
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        original = self.encoder(batch)
        noised = self.bnoise(batch, p=0.15)
        corrupted = self.encoder(noised)
        recovered = self.tfenc(einops.rearrange(corrupted, "s b d -> b s d"))
        recovered = einops.rearrange(recovered, "b s d -> s b d")
        return original, corrupted, recovered
