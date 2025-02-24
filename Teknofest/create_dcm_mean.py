import pandas as pd

import numpy as np
import pydicom

# Read CSV files
df = pd.read_csv("../Teknofest2024_final.csv")

means = [0] * len(df["path"])
stds = [0] * len(df["path"])

root = "../"
relative_path = "Teknofest_2024/"


print(len(df))
for i, path in enumerate(df["path"]):
    try:
        if i % 100 == 0:
            print(i)
        if df["year"][i] != 24:
            means[i] = df["dcm_mean"][i]
            stds[i] = df["dcm_std"][i]
            continue
        path = path.replace("png", "dcm")
        ds = pydicom.read_file(path)
        img = ds.pixel_array
        mean = img.mean()
        std = img.std()
        means[i] = mean
        stds[i] = std
        print(path)
    except Exception as e:
        print(e)
        print(path)

df["dcm_mean"] = means
df["dcm_std"] = stds

df.to_csv("../Teknofest2024_final.csv", index=False)

print("Done")