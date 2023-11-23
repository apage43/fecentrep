from pathlib import Path
import fer.data as fecdata
import torch.nn.functional as F
from fer.model import Config, TabDataset, TabularDenoiser
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lrsched
import math
from fer.multitask import UncertaintyWeightedLoss
import fire
from tqdm import tqdm
import wandb
from dataclasses import asdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

device = "cuda:1"

df = fecdata.pac_to_pac_transactions()
dataset, df, labelers = fecdata.prepare(df)


cfg = Config(
    embedding_init_std=1 / 512.0,
    tied_encoder_decoder_emb=True,
    entity_emb_normed=False,
    cos_sim_decode_entity=False,
    transformer_dim=384,
    transformer_heads=12,
    transformer_layers=8,
    entity_dim=384,
)
lr = 1e-3
n_epochs = 4
model = TabularDenoiser(
    cfg,
    n_entities=max(dataset["src"].max(), dataset["dst"].max()) + 1,
    n_etype=dataset["etype"].max() + 1,
    n_ttype=dataset["ttype"].max() + 1,
)
tds = TabDataset(dataset)

# %%

model = model.to(device)
model = torch.compile(model)

# %%
splitgen = torch.Generator().manual_seed(41)
batch_size = 3000
train_set, val_set = random_split(tds, [0.9, 0.1], generator=splitgen)
tdl = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    # persistent_workers=True,
)
vdl = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    # persistent_workers=True,
)

# %%


class WarmupConstantSchedule(lrsched.LambdaLR):
    """Linear warmup and then constant.
    Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
    Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.0


class WarmupCosineSchedule(lrsched.LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


dtsks = sorted(k for k in dataset.keys() if k.startswith("scaled_dt_"))


def decoder_loss(encoded, batch):
    srclogits, dstlogits, etlogits, ttlogits, amtd, amtpos, dt_pred = model.decoder(
        encoded, model.encoder
    )
    srcloss = F.cross_entropy(srclogits, batch["src"].squeeze())
    dstloss = F.cross_entropy(dstlogits, batch["dst"].squeeze())
    etloss = F.cross_entropy(etlogits, batch["etype"].squeeze())
    ttloss = F.cross_entropy(ttlogits, batch["ttype"].squeeze())
    amtloss = F.mse_loss(amtd, batch["amt"])
    amtposloss = F.binary_cross_entropy_with_logits(
        amtpos, batch["amt_pos"].to(torch.float)
    )
    # print(dt_pred.shape)
    dt_targets = torch.cat([batch[k].squeeze(dim=1) for k in dtsks], dim=1)
    # print(dt_targets.shape)
    dt_loss = F.mse_loss(dt_pred, dt_targets)
    return dict(
        srcloss=srcloss,
        dstloss=dstloss,
        etloss=etloss,
        ttloss=ttloss,
        amtloss=amtloss,
        amtposloss=amtposloss,
        dt_loss=dt_loss,
    )


def train():
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = WarmupCosineSchedule(optimizer, 1000, t_total=len(tdl) * n_epochs)
    n_losses = 14
    # lossweighter = CoVWeightingLoss(n_losses)
    lossweighter = UncertaintyWeightedLoss(n_losses)
    torch.set_float32_matmul_precision("high")
    with wandb.init(
        project="fecentrep2", save_code=True, config=dict(lr=lr, **asdict(cfg))
    ) as run:
        for epoch in range(n_epochs):
            with tqdm(tdl) as t:
                for i, batch in enumerate(t):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    model.zero_grad()
                    orig, corrupted, recovered = model(batch)
                    enclosses = decoder_loss(orig, batch)
                    reclosses = decoder_loss(recovered, batch)
                    # distloss = F.mse_loss(orig, recovered)
                    # margin = 0.1
                    # ocdiff = (orig != corrupted).max(dim=2).values.max(dim=0).values.float()
                    # rec_corrupt_err = ((recovered-corrupted).pow(2).mean(dim=2).mean(dim=0) * ocdiff).sum() / ocdiff.sum()
                    # repel_loss = F.relu(margin - rec_corrupt_err)
                    all_losses = {}
                    all_losses.update({f"enc/{k}": v for k, v in enclosses.items()})
                    all_losses.update({f"rec/{k}": v for k, v in reclosses.items()})
                    # all_losses['dist_loss'] = distloss
                    # all_losses['repel_loss'] = repel_loss
                    weighted_loss = lossweighter.forward(
                        [lv for _, lv in sorted(all_losses.items())]
                    )
                    total_loss = weighted_loss
                    all_losses["total_loss"] = total_loss
                    wandb.log(dict(**all_losses, lr=scheduler.get_last_lr()[0]))
                    total_loss.backward()
                    t.set_postfix(dict(loss=str(total_loss)))
                    optimizer.step()
                    scheduler.step()
        torch.save(model.state_dict(), f"{run.name}.bin")
    wandb.finish()


def upload_atlas(filename: str, do_norm=True):
    model.load_state_dict(torch.load(filename))
    entemb = model.encoder.entity_embeddings.weight.detach().cpu().numpy()
    print(entemb.shape)
    id2cid = labelers["id_labeler"].encoder.classes_
    idorder = pd.DataFrame({"CMTE_ID": id2cid})

    def read_frame(header_file, data_file, dtypes={}):
        header = pd.read_csv(header_file)
        dt = {c: str for c in header.columns}
        dt.update(dtypes)
        data = pd.read_csv(data_file, sep="|", names=header.columns, dtype=dt)
        return data

    def read_cm(year, basedir="./data"):
        cm = read_frame(
            f"{basedir}/cm_header_file.csv",
            f"{basedir}/{year}/cm.txt",
            dtypes={
                c: "str"
                for c in (
                    "CMTE_DSGN",
                    "CMTE_TP",
                    "CMTE_PTY_AFFILIATION",
                    "CMTE_FILING_FREQ",
                )
            },
        )
        return cm

    cmdf = idorder.join(
        pd.concat([read_cm(2020), read_cm(2022), read_cm(2024)])
        .drop_duplicates(subset=["CMTE_ID"], keep="last")
        .set_index("CMTE_ID"),
        on="CMTE_ID",
    ).dropna(subset=["CMTE_NM"]).fillna('N/A')
    namedemb = entemb[cmdf.index]

    from nomic import atlas

    atlas.map_embeddings(
        normalize(namedemb) if do_norm else namedemb,
        data=cmdf.reset_index(drop=True),
        name="fecentrep-2" + ("-norm" if do_norm else "") + f'-{Path(filename).stem}',
        colorable_fields=["CMTE_TP", "CMTE_DSGN", "ORG_TP", "CMTE_PTY_AFFILIATION"],
        id_field="CMTE_ID",
        topic_label_field="CMTE_NM",
        reset_project_if_exists=True,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "atlas": upload_atlas,
        }
    )
