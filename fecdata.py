
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client


def read_frame(header_file, data_file, dtypes={}, dask=False):
    header = pd.read_csv(header_file)
    dt = {c: str for c in header.columns}
    dt.update(dtypes)
    if dask:
        data = dd.read_csv(data_file, sep="|", names=header.columns, dtype=dt)
    else:
        data = pd.read_csv(data_file, sep="|", names=header.columns, dtype=dt)
    return data


def import_fec_data(basedir="./data", dc=None):
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
    for year in ["2020", "2022", "2024"]:
        yearcm, yearoth = read_cmoth(year)
        if cm is None:
            cm = yearcm
            oth = yearoth
        else:
            cm = pd.concat([cm, yearcm], axis=0)
            oth = dd.multi.concat([oth, yearoth], axis=0)
    cm = cm.groupby("CMTE_ID").last().fillna("N/A")

    for cat in ["CMTE_DSGN", "CMTE_TP", "CMTE_PTY_AFFILIATION", "CMTE_FILING_FREQ"]:
        cm = cm.assign(**{cat: cm[cat].astype("category")})

    interpac_txn = oth.dropna(
        subset=["TRANSACTION_DT", "OTHER_ID", "CMTE_ID", "ENTITY_TP", "TRANSACTION_TP"]
    ).categorize(
        columns=[
            "AMNDT_IND",
            "RPT_TP",
            "TRANSACTION_PGI",
            "TRANSACTION_TP",
            "ENTITY_TP",
        ]
    )
    interpac_txn = interpac_txn[interpac_txn["TRANSACTION_DT"] > "2018-10-01"]

    return cm, interpac_txn


def main():
    cm, interpac_txn = import_fec_data()
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

    interpac_txn.shuffle(["CMTE_ID", "OTHER_ID", "TRAN_ID"], npartitions=1).to_parquet(
        "./interpactx.parquet",
        write_index=False,
        overwrite=True,
        compression="zstd",
    )
    cm.to_parquet("./cm.parquet")
    print("Done")


if __name__ == "__main__":
    main()
