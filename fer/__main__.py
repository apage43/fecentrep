from pathlib import Path

import pandas as pd



def fer():
    print(Path.cwd())
    itoth = pd.read_parquet("./interpactx.parquet/part.0.parquet")
    print(itoth)
    ...


if __name__ == "__main__":
    fer()
