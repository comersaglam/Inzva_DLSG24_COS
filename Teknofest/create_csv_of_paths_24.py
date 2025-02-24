import pandas as pd
import os

inputdir = "../Teknofest_2024/DCM/"
paths = []

inputfilelist = []
for i in range(10001, 12722):
    inputfilelist.append(str(i))

for infpath1 in inputfilelist:
    infpath = os.path.join(inputdir, infpath1)
    if not (os.path.isdir(infpath)):
        continue

    for infile in os.listdir(infpath):
        infilepath = os.path.join(infpath, infile)
        paths.append(infilepath)

df = pd.DataFrame(paths, columns=["paths"])
df.to_csv("../Teknofest2024_paths.csv", index=True)

print("Done")