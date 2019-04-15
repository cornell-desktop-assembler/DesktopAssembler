import pickle

import numpy as np

with open("data.pkl", "rb") as fin:
    data = pickle.load(fin)

sku2i, i2sku, i2rgb, raw, reduced = data["sku2i"], data["i2sku"], data["i2rgb"], data["raw"], data["reduced"]

raw_normalized = raw / 255

u, s, vt = np.linalg.svd(raw_normalized, full_matrices=True)

after_svd = u[:, :2]

def euclidean_distances()