import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import common

from plotting.plot_input_features import *
import scipy.optimize as spo
import scipy.stats as sps
import glob

def get_pres(sig_proc):
  """Return the correct preselection range for this sig proc"""
  if ("XToHHggTauTau" in sig_proc) or ("HHTo2G2Tau" in sig_proc): #if graviton
    return (100, 180)
  elif "NMSSM_XYH_Y_gg_H_tautau" in sig_proc: #if Y_gg
    if common.LOW_MASS_MODE:
      return (65, 150)
    else:
      return (100, 1000)
  elif "NMSSM_XYH_Y_tautau_H_gg" in sig_proc: #if Y_tautau
    return (100, 180)

def plot_2d(bkg, score_column, save_path):
  mgg = bkg.Diphoton_mass
  score = bkg[score_column]

  plt.hist2d(mgg,score, bins=50, weights=bkg.weight)
  plt.colorbar()
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel(score_column)
  plt.savefig("%s.png"%save_path)
  plt.close()

# def exp(x, args):
#   return args[0]*np.exp(-args[1]*((x-100)/100))
  #return args[0]*np.exp(-args[1]*((x)/100))

# # def exp_int(args, a, b):
# #   return -(100/args[1]) * (exp(b, args) - exp(a, args))

import scipy.integrate as spi
def exp_int(args, a, b):
  if type(b) == float or type(b) == int:
    return spi.quad(exp, a, b, args=args)[0]
  else:
    return np.array([ spi.quad(exp, a, bi, args=args)[0] for bi in b ])

def exp(x, args):
  return args[0] * np.power(x/100, -args[1])

# def exp(x, args):
#   return args[0]*np.exp(-args[1]*((x)/100)) + args[2]*np.exp(-args[3]*((x)/100))

def chi2(args, x, y, y_err):
  y_pred = exp(x, args)
  return (((y-y_pred)/y_err)**2).sum() / (len(x)-2)

def do_fit(x, y, y_err):
  y_err[y_err==0] = min(y_err[y_err!=0])

  N_guess = y[0] / 2
  res = spo.minimize(chi2, [N_guess, 0.5], args=(x, y, y_err), bounds=((0, sum(y)), (0, 10)), method="Nelder-Mead")
  #res = spo.minimize(chi2, [N_guess, 0.1, N_guess, 0.9], args=(x, y, y_err), bounds=((0, sum(y)), (0, 10), (0, sum(y)), (0, 10)), method="SLSQP")
  print(res.x)

  #return lambda x: exp(x, res.x), chi2(res.x, x, y, y_err)
  return res.x, chi2(res.x, x, y, y_err)

def get_grad(x, y):
  h = x[1] - x[0] # assume constant step size
  return (y[1:] - y[:-1]) / h

def get_grad_err(x, y, y_err):
  h = x[1] - x[0] # assume constant step size
  y_high = y[1:]
  y_low = y[:-1]
  grad = (y_high - y_low) / h

  y_high_var = y[1:] + y_err[1:]
  y_low_var = y[:-1] - y_err[:-1]
  grad_var = (y_high_var - y_low_var) / h

  return abs(grad - grad_var)

def get_grad2(x, y):
  h = x[1] - x[0] # assume constant step size
  return (y[2:] - 2*y[1:-1] + y[:-2]) / h**2

def plot_bkg(bkg, proc_dict, column, nbins, feature_range, save_path, auto_legend=True, title=None, mh=None):
  plt.rcParams["figure.figsize"] = (10,8)
  
  bkg_stack, bkg_stack_w, bkg_stack_labels = createBkgStack(bkg, column, proc_dict)
  for i in range(len(bkg_stack)):
    if bkg_stack_w[i].sum() == 0:
      bkg_stack_labels[i] = "_"+bkg_stack_labels[i]
        
  bkg_stack_ungroup, bkg_stack_w_ungroup, bkg_stack_labels_ungroup = createBkgStack(bkg, column, proc_dict, group=False)

  n, edges, patches = plt.hist(bkg_stack, nbins, range=feature_range, weights=bkg_stack_w, label=bkg_stack_labels, stacked=True, color=colour_schemes[len(bkg_stack)], zorder=7) #background
  bin_centres = (edges[:-1]+edges[1:])/2

  bkg_sumw, bkg_error = getBkgError(bkg_stack_ungroup, bkg_stack_w_ungroup, edges)

  plt.fill_between(edges, np.append(bkg_sumw-bkg_error, 0), np.append(bkg_sumw+bkg_error, 0), step="post", alpha=0.5, color="grey", zorder=8) #background uncertainty
  plt.ylabel("Events")

  args, chi2 = do_fit(bin_centres, bkg_sumw, bkg_error)
  bkg_f = lambda x: exp(x, args)
  pval = 1-sps.chi2.cdf(chi2, 1)
  fit_label = "Power law fit\n" + r"$\chi^2 = %.2f, p_{\mathrm{value}} = %.2f$"%(chi2, pval)
  plt.plot(bin_centres, bkg_f(bin_centres), label=fit_label, zorder=10)

  # bkg_cdf = lambda x: exp_int(args, feature_range[0], x) / exp_int(args, feature_range[0], feature_range[1])
  # ks_res = sps.kstest(bkg.Diphoton_mass, bkg_cdf)
  # plt.plot(bin_centres, bkg_f(bin_centres), label=f"chi2 = {chi2:.2f}, pval={pval:.2f}, ks_pval={ks_res.pvalue:.2f}", zorder=10)

  #mplhep.cms.label(llabel="Work in Progress", data=True, lumi=138, loc=0)

  plt.xlabel(r"$m_{\gamma\gamma}$ [GeV]")
  plt.legend()
  #plt.title(title)
  # if mh is not None:
  #   h = plt.ylim()[1]/5
  #   plt.arrow(mh, plt.ylim()[1], 0, -h, width=2, head_width=4, head_length=h/4, zorder=10)
  #plt.yscale("log")
  plt.tight_layout()
  plt.savefig("%s.png"%(save_path))
  with open("%s_pval.txt"%save_path, "w") as f:
    f.write("%.3f"%pval)
  #plt.savefig("%s_log.png"%(save_path))
  plt.clf()

def makeSummary(outdir):
  pval_files = glob.glob(f"{outdir}/*_pval.txt")
  pvals = []
  for fname in pval_files:
    with open(fname, "r") as f:
      pvals.append(float(f.read()))
  
  ordering = np.argsort(pvals)
  pval_files = np.array(pval_files)[ordering]
  pvals = np.array(pvals)[ordering]
  
  with open("%s/summary.txt"%outdir, "w") as f:
    for fname, pval in zip(pval_files, pvals):
      proc = fname.split("/")[-1]
      f.write(f"{proc} {pval}\n")
  
def main(args):
  columns = common.getColumns(args.parquet_input)

  if args.score is None:
    args.score = list(filter(lambda x: "intermediate" in x, columns))

  if args.batch:
    if not args.batch_split:
      common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True
    else:
      score_copy = args.score.copy()
      for score in score_copy:
        args.score = [score]
        common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
      return True   

  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  all_bkg_ids = [proc_dict[proc] for proc in common.bkg_procs['all']]

  gjet_ids = [proc_dict[proc] for proc in common.bkg_procs['GJets']]
  smhiggs_ids = [proc_dict[proc] for proc in common.bkg_procs['SM Higgs']]
  ttbar_jets_ids = [proc_dict["TTJets"]]
  diphoton_ids = [proc_dict[proc] for proc in common.bkg_procs["Diphoton"]]
  
  
  bkg_ids = set(all_bkg_ids) - set(smhiggs_ids) 
  if args.exclude_gjet:
    bkg_ids = bkg_ids - set(gjet_ids) - set(ttbar_jets_ids)
  if args.only_diphoton:
    bkg_ids = set(diphoton_ids)
    
  os.makedirs(args.outdir, exist_ok=True)

  first_col = True
  for column in args.score:
    columns_to_read = ["Diphoton_mass", "weight", "process_id"] + [column]
    df = pd.read_parquet(args.parquet_input, columns=columns_to_read)
    df = df[df.process_id.isin(bkg_ids)]
      
    if "intermediate" in column:
      sig_proc = column.split("score_")[1]
      print(sig_proc)
      mx, my = common.get_MX_MY(sig_proc)
      # pres = list(get_pres(sig_proc))
      # # if my-100 > pres[0]:
      # #   pres[0] = my-100
      # # if my+100 < pres[1]:
      # #   pres[1] = my+100
      # # pres[1] = 400
      # #print(sig_proc, mx, my, pres)
      # #pres = [100, 300]
      # nbins = int((pres[1]-pres[0]) / 16)
      nbins = 10

      def processDF(df, column, threshold, proc_dict, mh, reverse_threshold=False, num_events=False):
        if num_events:
          df_col_sorted = df[column].sort_values(ascending=False)
          sumw_cdf = df.loc[df_col_sorted.index, "weight"].cumsum()
          sumw2_cdf = (df.loc[df_col_sorted.index, "weight"]**2).cumsum()
          frac_uncert = np.sqrt(sumw2_cdf) / sumw_cdf
          idx = frac_uncert > threshold
                    
          df_cut = df.loc[idx]
          print(df_cut[column])
        else:
          if reverse_threshold:
            df_cut = df[df[column] < threshold]
          else:
            df_cut = df[df[column] > threshold]
        # quant = list(df_cut[df.process_id==proc_dict["DiPhoton"]].Diphoton_mass.quantile([0.0, 0.95]))
        # quant[0] = 100
        # if quant[1] < (mh+10):
        #   quant[1] = mh + 10
        quant = [100, 180]
        #quant = [80, 150]
        df_cut = df_cut[(df_cut.Diphoton_mass > quant[0]) & (df_cut.Diphoton_mass < quant[1])]
        print(quant)
        return df_cut, quant

      mh = my
      #df_cut = df[df[column] < 0.01]
      # df_cut, pres = processDF(df, column, 0.01, proc_dict, mh, reverse_threshold=True)
      # plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_lt0p01"%column), title=f"MX={mx}, MY={my}: score < 0.01", mh=mh)

      # df_cut, pres = processDF(df, column, 0.95, proc_dict, mh, reverse_threshold=True)
      # plot_bkg(df_cut, proc_dict, "Diphoton_mass", 40, pres, os.path.join(args.outdir, "%s_lt0p01"%column), title=f"MX={mx}, MY={my}: score < 0.95", mh=mh)

      #df_cut = df[df[column] > 0.99]
      # df_cut, pres = processDF(df, column, 0.95, proc_dict, mh)
      # plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_gt0p99"%column), title=f"MX={mx}, MY={my}: score > 0.99", mh=mh)

      # #df_cut = df[df[column] > 0.999]
      # df_cut, pres = processDF(df, column, 0.999, proc_dict, mh)
      # plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_gt0p999"%column), title=f"MX={mx}, MY={my}: score > 0.999", mh=mh)
      
      df_cut, pres = processDF(df, column, 0.05, proc_dict, mh, num_events=True)
      plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_frac_uncert_0.05"%column), title=r"$(m_X, m_Y) = %d, %d\, $GeV, $\sqrt{\sum w^2} / \sum w$ = 0.05"%(mx,my), mh=mh)
      
      df_cut, pres = processDF(df, column, 0.1, proc_dict, mh, num_events=True)
      plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_frac_uncert_0.1"%column), title=f"MX={mx}, MY={my}: frac uncert = 0.1", mh=mh)

      # df_cut, pres = processDF(df, column, 0.5, proc_dict, mh)
      # plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_gt0p5"%column), title=f"MX={mx}, MY={my}: score > 0.5", mh=mh)

      # df_cut, pres = processDF(df, column, 0.5, proc_dict, mh, reverse_threshold=True)
      # plot_bkg(df_cut, proc_dict, "Diphoton_mass", nbins, pres, os.path.join(args.outdir, "%s_lt0p5"%column), title=f"MX={mx}, MY={my}: score < 0.5", mh=mh)

  makeSummary(args.outdir)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)
  parser.add_argument('--batch-split', action="store_true")
  parser.add_argument('--score', type=str, nargs="+", default=None)
  parser.add_argument('--exclude-gjet', action="store_true")
  parser.add_argument('--only-diphoton', action="store_true")

  args = parser.parse_args()
  
  df = main(args)