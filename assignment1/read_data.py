import pandas as pd

# Se serve, imposta la working directory (equivalente di setwd in R)
# import os
# os.chdir("/home/pbac/g/course02417/2025/assignment1")

# Read training data
D = pd.read_csv("DST_BIL54.csv")

# Equivalent di str(D) in R
print(D.info())
print(D.head())

# Parse time: in R era "YYYY-MM" e aggiungevi "-01"
# tz="UTC" -> usiamo datetime timezone-aware
D["time"] = pd.to_datetime(D["time"].astype(str) + "-01", format="%Y-%m-%d", utc=True)

print(D["time"])
print(D["time"].dtype)

# Year to month: 1900 + year + mon/12 in R (POSIXlt$year è "anni dal 1900")
# In pandas: year reale + (month-1)/12
D["year"] = D["time"].dt.year + (D["time"].dt.month - 1) / 12.0

# Make output variable floating point and scale
D["total"] = pd.to_numeric(D["total"], errors="coerce") / 1e6

# Divide into train and test set
teststart = pd.Timestamp("2024-01-01", tz="UTC")
Dtrain = D[D["time"] < teststart].copy()
Dtest  = D[D["time"] >= teststart].copy()

print("Train shape:", Dtrain.shape)
print("Test shape:", Dtest.shape)
Dtrain.to_csv("train.csv", index=False)
Dtest.to_csv("test.csv", index=False)

