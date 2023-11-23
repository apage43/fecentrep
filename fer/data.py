import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .encoding import IdentifierLabeler


def pac_to_pac_transactions():
    txdf = pd.read_parquet("interpactx.parquet")
    return txdf


def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix
    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    if time:
        attr = attr + ["Hour", "Minute", "Second"]
    # Pandas removed `dt.week` in v1.1.10
    week = (
        field.dt.isocalendar().week.astype(field.dt.day.dtype)
        if hasattr(field.dt, "isocalendar")
        else field.dt.week
    )
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower()) if n != "Week" else week
    mask = ~field.isna()
    df[prefix + "Elapsed"] = np.where(
        mask, field.values.astype(np.int64) // 10**9, np.nan
    )
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


def prepare(df: pd.DataFrame):
    labelers = dict(
        id_labeler=IdentifierLabeler(cols=["CMTE_ID", "OTHER_ID"]),
        etype_labeler=IdentifierLabeler(cols=["ENTITY_TP"]),
        ttype_labeler=IdentifierLabeler(cols=["TRANSACTION_TP"]),
    )

    for labeler in labelers.values():
        df = labeler.fit_transform(df)

    df["amt_positive"] = df["TRANSACTION_AMT"] >= 0
    df["amt_absolute"] = df["TRANSACTION_AMT"].abs()

    amtscaler = StandardScaler()

    # i want to predict rough order of magnitude more than the exact dollar amount
    df = df.assign(
        amt_scaled=amtscaler.fit_transform(np.log10(df[["amt_absolute"]].values + 1))
    )
    df = add_datepart(df, "TRANSACTION_DT", prefix="")

    dataset = dict(
        src=df["CMTE_ID"].values,
        dst=df["OTHER_ID"].values,
        etype=df["ENTITY_TP"].values,
        ttype=df["TRANSACTION_TP"].values,
        amt=df["amt_scaled"].values,
        amt_pos=df["amt_positive"].values,
        dt_year=df["Year"].values,
        dt_month=df["Month"].values,
        dt_week=df["Week"].values,
        dt_day=df["Day"].values,
        dt_weekday=df["Dayofweek"].values,
        dt_yearday=df["Dayofyear"].values,
    )

    dtscalers = {k: StandardScaler() for k in dataset.keys() if k.startswith("dt_")}
    for k in dtscalers.keys():
        scaled = dtscalers[k].fit_transform(dataset[k].reshape(-1, 1))
        dataset[f"scaled_{k}"] = scaled
    return (dataset, df, labelers)
