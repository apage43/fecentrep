import math
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import re


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

def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a

def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


def import_fec_data(basedir = "./data", dc=None):
    if not dc:
        dc = Client()
    print(dc)
    def read_cmoth(year):
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
        ).set_index("CMTE_ID")

        oth = read_frame(
            f"{basedir}/oth_header_file.csv",
            f"{basedir}/{year}/itoth.txt",
            dtypes={
                **{
                    c: "str"
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
        return cm, oth
    cm = None
    oth = None
    for year in ['2020','2022','2024']:
        yearcm, yearoth = read_cmoth(year)
        if cm is None:
            cm = yearcm
            oth = yearoth
        else:
            cm = cm.append(yearcm)
            oth = oth.append(yearoth)
    cm = cm.groupby("CMTE_ID").last().fillna("N/A")

    for cat in ["CMTE_DSGN", "CMTE_TP", "CMTE_PTY_AFFILIATION", "CMTE_FILING_FREQ"]:
        cm = cm.assign(**{cat: cm[cat].astype("category")})

    interpac_txn = oth.dropna(
        subset=["TRANSACTION_DT", "OTHER_ID", "CMTE_ID", "ENTITY_TP", "TRANSACTION_TP"]
    ).categorize(columns=["AMNDT_IND", "RPT_TP", "TRANSACTION_PGI", "TRANSACTION_TP", "ENTITY_TP"])
    interpac_txn = interpac_txn[interpac_txn["TRANSACTION_DT"] > "2018-10-01"]

    return cm, interpac_txn

def main():
    cm, interpac_txn = import_fec_data()
    origoth = interpac_txn
    interpac_txn = interpac_txn[
        [
            "CMTE_ID",
            "OTHER_ID",
            "ENTITY_TP",
            "TRANSACTION_TP",
            "TRANSACTION_AMT",
            "TRANSACTION_DT",
        ]
    ]

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
    add_datepart(sdf, 'TRANSACTION_DT', 'txd_', drop=False)
    sdf[dayscol] = (sdf["TRANSACTION_DT"] - pd.to_datetime(basedate)).dt.days
    #amtscale = StandardScaler()
    logamtscale = StandardScaler()
    #dayscale = StandardScaler()
    sdf["amt_positive"] = sdf["TRANSACTION_AMT"] > 0
    sdf["amt_abs"] = sdf["TRANSACTION_AMT"].apply(np.abs)
    sdf["amt_log1p"] = sdf["amt_abs"].apply(np.log1p) # prolly more useful to try and get the magnitude right than the precise amount???
    # sdf["amt_scaled"] = amtscale.fit_transform(sdf[["amt_abs"]].values)
    sdf["amt_log_scaled"] = logamtscale.fit_transform(sdf[["amt_log1p"]].values)
    #sdf["time_abs_scaled"] = dayscale.fit_transform(sdf[[dayscol]].values)
    # "fourier features" to encode datetime
    fourfs = []
    idt = sdf[dayscol]

    def add_ft(pfx, scl, fn, c, idt):
        feat = f"{pfx}_{scl}{c}"
        sdf[feat] = fn(idt.values * math.pi * 1.0/scl)
        fourfs.append(feat)

    def add_fourfs(pfx, col, scls=[2920, 365, 90, 30, 7, 1]):
        for scl in scls:
            add_ft(pfx, scl, np.sin, "s", col)
            add_ft(pfx, scl, np.cos, "c", col)

    add_fourfs("wk", sdf['txd_Week'], [52])
    add_fourfs("mo", sdf['txd_Month'], [12])
    add_fourfs("dow", sdf['txd_Dayofweek'], [7])
    add_fourfs("dom", sdf['txd_Day'], [31])
    add_fourfs("yr", sdf['txd_Year'], [10])
    add_fourfs("wom", sdf['txd_Week'], [5])

    bincols = ["amt_positive"]
    contcols = ["amt_log_scaled"] + fourfs
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

    sdf.to_parquet("./fecpreprocd.parquet")
    origoth.shuffle(['CMTE_ID','OTHER_ID','TRAN_ID'], npartitions=8).to_parquet(
        "./interpactx.parquet", write_index=False, overwrite=True, compression='zstd',
    )
    cm.to_parquet("./cm.parquet")
    print("Done")


if __name__ == "__main__":
    main()
