import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def getLimits(results_path):
  with open(results_path, "r") as f:
    results = f.readlines()

  masses = []
  for line in results:
    m = line.split(".")[0].split("_")[-1]
    mx = int(m.split("mx")[1].split("my")[0])
    my = int(m.split("my")[1].split("mh")[0])
    if "mh" in m: mh = int(m.split("mh")[1])
    else:         mh = 125
    if [mx, my, mh] not in masses:
      masses.append([mx, my, mh])

  limits = np.zeros((5, len(masses)))
  limits_no_sys = np.zeros((5, len(masses)))
  limits_no_res_bkg = np.zeros((5, len(masses)))
  limits_no_dy_bkg = np.zeros((5, len(masses)))

  for line in results:
    m = line.split(".")[0].split("_")[-1]
    mx = int(m.split("mx")[1].split("my")[0])
    my = int(m.split("my")[1].split("mh")[0])
    if "mh" in m: mh = int(m.split("mh")[1])
    else:         mh = 125
    idx1 = masses.index([mx, my, mh])
    if "2.5%" in line:
      idx2=0
    elif "16.0%" in line:
      idx2=1
    elif "50.0%" in line:
      idx2=2
    elif "84.0%" in line:
      idx2=3
    elif "97.5%" in line:
      idx2=4
    
    limit = float(line.split("r < ")[1])

    if "no_sys" in line:
      limits_no_sys[idx2][idx1] = limit
    elif "no_res_bkg" in line:
      limits_no_res_bkg[idx2][idx1] = limit
    elif "no_dy_bkg" in line:
      limits_no_dy_bkg[idx2][idx1] = limit
    else:
      limits[idx2][idx1] = limit

  #print(limits[2])
  #print(limits_no_sys[2])

  masses = np.array(masses)
  #sort out scan over mh (mgg)
  if len(np.unique(np.array(masses)[:,2])) != 1: #if more than 125 in mh
    #find places where mx and mh overlap
    for mx in np.unique(masses[:,0]):
      uniques, counts = np.unique(masses[masses[:,0]==mx, 2], return_counts=True)
      assert sum(counts>2) == 0 #should not have more than 1 overlap
      overlap_mh = uniques[counts==2]

      for mh in overlap_mh:
        idx1, idx2 = np.where( (masses[:,0]==mx) & (masses[:,2]==mh) )[0]
        if limits[2][idx1] < limits[2][idx2]:
          to_delete = idx2
        else:
          to_delete = idx1
        masses = np.delete(masses, to_delete, axis=0)
        limits = np.delete(limits, to_delete, axis=1)
        limits_no_sys = np.delete(limits_no_sys, to_delete, axis=1)
        limits_no_res_bkg = np.delete(limits_no_res_bkg, to_delete, axis=1)
        limits_no_dy_bkg = np.delete(limits_no_dy_bkg, to_delete, axis=1)

    masses[:,1] = masses[:,2] #set my to be mh
  
  #masses = masses[:,:2]
  #print(masses)

  return masses, limits, limits_no_sys, limits_no_res_bkg, limits_no_dy_bkg

if __name__=="__main__":
  limit_vs_N_dir = sys.argv[1]
  limits_N = []
  
  masses = None
  N = [n for n in range(4, 21, 2)]
  for n in N:
    all_limits = getLimits(os.path.join(limit_vs_N_dir, str(n), "Combine", "Results", "combine_results_summary.txt"))
    if masses is None:
      masses = all_limits[0]
    else:
      assert (masses == all_limits[0]).all()

    limits_N.append(all_limits[1])

  limits_N = np.array(limits_N)[:,2,:]

  for i, m in enumerate(masses):
    plt.plot(N, limits_N[:,i]/limits_N[3,i], label="mx%dmy%d"%(m[0],m[1]))

  plt.xlabel(r"$N_{sidebands}$")
  plt.ylabel("Expected 95% CL Limit [fb] / Expected at N=10")
  plt.legend()
  plt.savefig("limit_vs_N_high_mass.png")