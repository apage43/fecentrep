import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pickle

class IdentifierSpaceLabeler:
    def __init__(self, cols):
        self.encoder = LabelEncoder()
        self.cols = cols

    def fit(self, df):
        self.encoder.fit(pd.concat(df[col] for col in self.cols))
        return self

    def transform(self, df):
        df = df.sample(frac=1.0)
        for col in self.cols:
            df[col] = self.encoder.transform(df[col])
        return df

    def fit_transform(self, df, **fp):
        self.fit(df)
        return self.transform(df)


def read_frame(header_file, data_file, dtypes={}, dask=False):
    header = pd.read_csv(header_file)
    dt = {c: str for c in header.columns}
    dt.update(dtypes)
    if dask:
        data = dd.read_csv(data_file, sep="|", names=header.columns, dtype=dt)
    else:
        data = pd.read_csv(data_file, sep="|", names=header.columns, dtype=dt)
    return data


def import_fec_data():
    dc = Client()
    print(dc)
    basedir = "./data/2022"
    cm = read_frame(
        f"{basedir}/cm_header_file.csv",
        f"{basedir}/cm.txt",
        dtypes={
            c: "category"
            for c in (
                "CMTE_DSGN",
                "CMTE_TP",
                "CMTE_PTY_AFFILIATION",
                "CMTE_FILING_FREQ",
            )
        },
    ).set_index("CMTE_ID")

    oth = read_frame(
        f"{basedir}/oth_header_file.csv",
        f"{basedir}/itoth.txt",
        dtypes={
            **{
                c: "category"
                for c in (
                    "AMNDT_IND",
                    "RPT_TP",
                    "TRANSACTION_PGI",
                    "TRANSACTION_TP",
                    "ENTITY_TP",
                )
            },
            **{"TRANSACTION_AMT": int},
        },
        dask=True,
    )
    oth["TRANSACTION_DT"] = dd.to_datetime(
        oth["TRANSACTION_DT"], format=r"%m%d%Y", errors="coerce"
    )

    interpac_txn = oth[
        [
            "CMTE_ID",
            "OTHER_ID",
            "ENTITY_TP",
            "TRANSACTION_TP",
            "TRANSACTION_AMT",
            "TRANSACTION_DT",
        ]
    ].dropna(
        subset=["TRANSACTION_DT", "OTHER_ID", "CMTE_ID", "ENTITY_TP", "TRANSACTION_TP"]
    )
    interpac_txn = interpac_txn[interpac_txn["TRANSACTION_DT"] > "2018-10-01"]

    print(f"itoth.txt (pac to pac transactions) has {len(interpac_txn)} rows")

    labelers = {
        "entities": IdentifierSpaceLabeler(["CMTE_ID", "OTHER_ID"]),
        "entity_types": IdentifierSpaceLabeler(["ENTITY_TP"]),
        "transaction_types": IdentifierSpaceLabeler(["TRANSACTION_TP"]),
    }

    ipdf = interpac_txn.compute()
    sdf = ipdf
    for _, l in labelers.items():
        sdf = l.fit_transform(sdf)
    sdf

    basedate = "2021-11-01"
    dayscol = f"tx_days_{basedate[:-5]}"
    sdf[dayscol] = (
        sdf["TRANSACTION_DT"] - pd.to_datetime(basedate)
    ).dt.days
    amtscale = StandardScaler()
    dayscale = StandardScaler()
    sdf["amt_positive"] = sdf["TRANSACTION_AMT"] > 0
    sdf["amt_abs"] = sdf["TRANSACTION_AMT"].apply(np.abs)
    sdf["amt_scaled"] = amtscale.fit_transform(sdf[["amt_abs"]].values)
    sdf["time_abs_scaled"] = dayscale.fit_transform(sdf[[dayscol]].values)
    # "fourier features" to encode datetime
    fourfs = []
    idt = sdf[dayscol].values

    def add_ft(scl, fn, c):
        feat = f"dt_{scl}{c}"
        sdf[feat] = fn(idt)
        fourfs.append(feat)

    for scl in [365, 90, 30, 7, 1]:
        add_ft(scl, np.sin, "s")
        add_ft(scl, np.cos, "c")

    bincols = ["amt_positive"]
    contcols = ["amt_scaled", "time_abs_scaled"] + fourfs
    entcols = ["CMTE_ID", "OTHER_ID"]
    catcols = ["ENTITY_TP", "TRANSACTION_TP"]

    meta = {
        "ents": labelers["entities"].encoder.classes_,
        "etype": labelers["entity_types"].encoder.classes_,
        "txtype": labelers["transaction_types"].encoder.classes_,
        "cols": {
            "conts": contcols,
            "cats": catcols,
            "bins": bincols,
            "ents": entcols,
        },
    }
    with open("meta.pkl", "wb") as of:
        pickle.dump(meta, of)

    sdf.to_parquet('./fecpreprocd.parquet')



if __name__ == '__main__':
    import_fec_data()