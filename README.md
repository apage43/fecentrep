# PAC embeddings w/ denoising autoencoder

Uses the 'Any transaction from one committee to another' dataset from the [FEC's bulk data page](https://www.fec.gov/data/browse-data/?tab=bulk-data), expects `itoth.txt`, `cm.txt`, `cm_header_file.csv`, `oth_header_file.csv` to exist in `./data`

requires `pytorch`, `pytorch-lightning`, `umap-learn`, `dask`, `fastparquet`

* run `fecdata.py` to generate `fecpreprocd.parquet` and `meta.pkl`
* run `interpac_embedding.py` to train embeddings

`eview.ipynb` can generate this [UMAP](https://umap-learn.readthedocs.io/en/latest/) plot of the embeddings colored using the PAC metadata in `cm.txt`, which was *not* seen by the model

![](embedplot.png)