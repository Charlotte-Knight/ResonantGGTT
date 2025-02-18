import pandas as pd
import sys
import argparse
import common

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
    return True
    
  df = pd.read_parquet(args.parquet_input)

  df["dilep_leadpho_mass_mgg"] = df.dilep_leadpho_mass / df.Diphoton_mass
  df["dilep_leadpho_mass_rotate"] = df.dilep_leadpho_mass - 0.450*df.Diphoton_mass

  df.loc[df.dilep_leadpho_mass==-9, "dilep_leadpho_mass_mgg"] = -9
  df.loc[df.dilep_leadpho_mass==-9, "dilep_leadpho_mass_rotate"] = -9

  df.to_parquet(args.parquet_output)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--parquet-output', '-o', type=str, required=True)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)

  args = parser.parse_args()
  
  df = main(args)