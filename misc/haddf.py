import pandas as pd
import sys

output_path = sys.argv[1]
input_paths = sys.argv[2:]

dfs = []
print("Merging %d files"%(len(input_paths)))

for file_path in input_paths:
  dfs.append(pd.read_parquet(file_path))

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_parquet(output_path)