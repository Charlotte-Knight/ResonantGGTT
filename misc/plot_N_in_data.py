d = "Graviton"

with open(f"Outputs_Feb/{d}/CatOptim_MC_Fit/N_in_data.txt", "r") as f:
  out = f.readlines()

from compare_optimisation_approaches import plotHistogram

current_proc = None
N_pred_dict = {}
N_pred = []
for i, line in enumerate(out):
  if "Data" in line:
    proc = line.split("Data")[1].split("cat")[0]
    if proc == current_proc:
      N_pred.append(int(out[i+1]))
    else:
      if current_proc is not None:
        N_pred_dict[current_proc] = N_pred[:-1]
      current_proc = proc 
      N_pred = [int(out[i+1])]

N_pred_dict[current_proc] = N_pred[:-1]  

print(N_pred_dict)
for proc in N_pred_dict:
  print(proc)
  print(N_pred_dict[proc][:10])
  plotHistogram(N_pred_dict[proc][:100], 10, f"Outputs_Feb/{d}/CatOptim_MC_Fit/N_in_data_%s.png"%(proc))

  assert 0 not in N_pred_dict[proc][:10]
    
