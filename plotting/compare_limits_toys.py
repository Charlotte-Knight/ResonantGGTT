import pandas as pd
import sys

asymp = pd.read_csv(sys.argv[1])
toys = pd.read_csv(sys.argv[2])

diff = asymp["obs"] / toys["obs"]

toys["diff"] = diff
print(toys)
print(diff.mean(), diff.min(), diff.max())