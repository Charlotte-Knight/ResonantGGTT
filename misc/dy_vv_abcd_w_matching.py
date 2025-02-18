"""
Inputs:
1. parquet files with DY events and their pNN scores
  i. Veto applied
  ii. Veto inverted
2. A json containing the gen classification of those events
    This is taken from a gen-level study and each event is assigned
    0, 1 or 2 depending on whether it is non-peaking, peaking
    or inconclusive.

Outputs:
1. Check whether the electron + pixel veto efficiency is correlated with
    pNN score.
2. Assuming negligible correlation, use ABCD method to estimate DY peaking
"""

import pandas as pd
import json
import sys

def addEventClassification(df, gen_classification):
  classification_list = []
  for idx, row in df.iterrows():
    print(row.year, row.event)
    classification_list.append(gen_classification[str(int(row.year))][str(int(row.event))])
  df["gen_classification"] = classification_list
  return df

parquet_file = sys.argv[1]
parquet_file_inverted = sys.argv[2]
summary_json = sys.argv[3]
summary_json_inverted = sys.argv[4]
gen_classification_file = sys.argv[5]

df = pd.read_parquet(parquet_file)
df_inverted = pd.read_parquet(parquet_file_inverted)

with open(summary_json, "r") as f:
  proc_dict = json.load(f)["sample_id_map"]
df = df[df.process_id == proc_dict["DY"]]
with open(summary_json_inverted, "r") as f:
  proc_dict_inverted = json.load(f)["sample_id_map"]
df_inverted = df_inverted[df_inverted.process_id == proc_dict_inverted["DYinvEveto"]]

with open(gen_classification_file, "r") as f:
  gen_classification = json.load(f)

# check if any overlap between 2016 and 2016APV
k2016 = list(gen_classification["2016"].keys())
k2016APV = list(gen_classification["2016APV"].keys())
assert len(set(k2016+k2016APV)) == len(k2016) + len(k2016APV)
# merge 2016 and 2016 APV
gen_classification["2016"].update(gen_classification["2016APV"])
del gen_classification["2016APV"]

#print("df")
#df = addEventClassification(df, gen_classification)
print("df_inverted")
df_inverted = addEventClassification(df_inverted, gen_classification)