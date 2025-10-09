import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

train_data = [
    {
        "problem": [256, 256, 256, "fp32"],
        "candidates": [[32, 32, 16], [64, 64, 16], [128, 128, 16]],
        "latencies": [2.0, 1.2, 1.5],
    },
    {
        "problem": [512, 512, 512, "fp32"],
        "candidates": [[128, 128, 32], [256, 256, 32]],
        "latencies": [2.5, 2.0],
    },
    {
        "problem": [256, 512, 128, "fp16"],
        "candidates": [[32, 64, 16], [64, 128, 16]],
        "latencies": [1.1, 0.9],
    },
]

dtype_encoder = LabelEncoder()
dtypes = [d["problem"][3] for d in train_data]
dtype_encoder.fit(dtypes)

X_list, y_latency_list, group_list, candidate_list = [], [], [], []

for d in train_data:
    m, n, k, dt = d["problem"]
    num_candidates = len(d["candidates"])
    for tile, latency in zip(d["candidates"], d["latencies"]):
        X_list.append(
            [m, n, k, dtype_encoder.transform([dt])[0]] + tile
        )  # problem + candidate tile
        y_latency_list.append(latency)
    group_list.append(num_candidates)
    candidate_list.append(d["candidates"])

X = np.array(X_list)
y_latency = np.array(y_latency_list)
group = np.array(group_list)

dtrain = xgb.DMatrix(X, label=y_latency)
dtrain.set_group(group)

params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "max_depth": 6,
    "eval_metric": "ndcg",
}

bst = xgb.train(params, dtrain, num_boost_round=50)

# Notice that the test_problems are all seen data.
test_problems = [
    [256, 256, 256, "fp32", [[32, 32, 16], [64, 64, 16], [128, 128, 16]]],
    [256, 512, 128, "fp16", [[32, 64, 16], [64, 128, 16]]],
]

for prob in test_problems:
    m, n, k, dt, candidates = prob
    X_test = []
    for tile in candidates:
        X_test.append([m, n, k, dtype_encoder.transform([dt])[0]] + tile)
    X_test = np.array(X_test)
    dtest = xgb.DMatrix(X_test)
    scores = bst.predict(dtest)
    best_idx = np.argmin(scores)
    print(
        f"Problem {m}x{n}x{k} ({dt}) -> Best tile: {candidates[best_idx]} with predicted score {scores[best_idx]:.3f}"
    )
