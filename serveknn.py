from typing import List, Union
import faiss
import numpy as np
import pickle
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import normalize

with open("./meta.pkl", "rb") as mf:
    meta = pickle.load(mf)

embs = np.load("./embeddings.npy")
index = faiss.read_index("./knn.index")
id2idx = {id: idx for idx, id in enumerate(meta["ents"])}
idx2id = {idx: id for id, idx in id2idx.items()}

app = FastAPI()

class KNNResult(BaseModel):
    pacids: List[str]
    dists: List[float]

@app.get("/pac_id_knn")
def knn_search(pac_id: str, limit: int = 10) -> KNNResult:
    if ',' in pac_id:
        pac_ids = pac_id.split(',')
        idxes = [id2idx[pac_id] for pac_id in pac_ids]
    else:
        idxes = [id2idx[pac_id]]
    query = np.array([embs[idx] for idx in idxes]).mean(axis=0,keepdims=True)
    query = normalize(query, axis=1)
    dists, ids = index.search(query, limit)
    pacids = [idx2id[id] for id in ids[0]]
    return KNNResult(pacids=pacids, dists=dists[0].tolist())