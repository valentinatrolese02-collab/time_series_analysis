import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Se serve, imposta la working directory (equivalente di setwd in R)
# import os
# os.chdir("/home/pbac/g/course02417/2025/assignment1")

# Read training data
D = pd.read_csv("DST_BIL54.csv")

# Equivalent di str(D) in R
#print(D.info())

# Parse time: in R era "YYYY-MM" e aggiungevi "-01"
# tz="UTC" -> usiamo datetime timezone-aware
D["time"] = pd.to_datetime(D["time"].astype(str) + "-01", format="%Y-%m-%d", utc=True)

#print(D["time"])
#print(D["time"].dtype)

# Year to month: 1900 + year + mon/12 in R (POSIXlt$year è "anni dal 1900")
# In pandas: year reale + (month-1)/12


D["year"] = D["time"].dt.year + (D["time"].dt.month - 1) / 12.0

# Make output variable floating point and scale
D["total"] = pd.to_numeric(D["total"], errors="coerce") / 1e6


#print(D.head())
# Divide into train and test set
teststart = pd.Timestamp("2024-01-01", tz="UTC")
Dtrain = D[D["time"] < teststart].copy()
#Dtest  = D[D["time"] >= teststart].copy()
X = Dtrain["year"].values

#print("X:", X)
#print("X shape:", X.shape)

y = Dtrain["total"].to_numpy()          # heights
n = len(y)

x_labels = [f"x{i}" for i in range(1, n+1)]  # x1..xn
x_pos = np.arange(n)                         # 0..n-1 positions

plt.figure()
plt.bar(x_pos, y)

plt.xticks(x_pos, x_labels, rotation=90)  # rotate so labels fit
plt.xlabel("Observations (x1..x{})".format(n))
plt.ylabel("Total (millions)")
plt.title("Total values for training set")
plt.tight_layout()
#plt.show()
#print("Train shape:", Dtrain.shape)
##print("Test shape:", Dtest.shape)
#Dtrain.to_csv("train.csv", index=False)
#Dtest.to_csv("test.csv", index=False)


#exercise2

y3 = Dtrain["total"].iloc[:3].to_numpy()
x3 = Dtrain["year"].iloc[:3].to_numpy()
x3 = np.c_[np.ones(3), x3]   # adds a column of 1s in front



print(x3.shape)  # should be [2018. , 2018.0833..., 2018.1666...]
print(y3.shape)  # your first 3 totals (in millions)






