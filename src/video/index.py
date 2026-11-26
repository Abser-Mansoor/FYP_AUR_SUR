"""Gallery search (Phase 9 Part E): exact FAISS IndexFlatIP or approximate IndexHNSWFlat over L2-normalised templates (inner product = cosine).

    idx = GalleryIndex(templates, kind="flat" | "hnsw", m=32, ef_construction=200, ef_search=64)
    sims, rows = idx.search(queries, k)
"""
import faiss
import numpy as np


class GalleryIndex:
    def __init__(self, templates, kind="flat", m=32, ef_construction=200, ef_search=64):
        x = np.ascontiguousarray(templates, np.float32)
        d = x.shape[1]
        if kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)
        self.kind = kind
        self.index.add(x)

    def search(self, queries, k=1):
        return self.index.search(np.ascontiguousarray(queries, np.float32), k)

    def __len__(self):
        return self.index.ntotal
