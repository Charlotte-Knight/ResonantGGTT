import pandas as pd
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep 
#mplhep.style.use("CMS")
import numpy as np

bkg_procs = {
  'Diphoton': ['DiPhoton'],
  'GJets': ['GJets'],
  'TT': ['TTGG', 'TTGamma', 'TTJets'],
  'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
  'VGamma': ['WGamma', 'ZGamma']
}


bkg_procs["all"] = [proc for key in bkg_procs.keys() for proc in bkg_procs[key]]

# Read in dataframes
df = pd.read_parquet(sys.argv[1])
with open(sys.argv[2], "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

data = df[df.process_id==proc_dict["Data"]]
bkg_proc_ids = [proc_dict[bkg_proc] for bkg_proc in bkg_procs["all"]]
bkg = df[df.process_id.isin(bkg_proc_ids)]
sig_proc_ids = [proc_dict["HHggTauTau"] ]
sig = df[df.process_id.isin(sig_proc_ids)]

# List of quantiles
n_quantiles = 10
quantiles = np.linspace(0,1,n_quantiles+1)
btag_quantiles = list(sig[sig.b_jet_1_btagDeepFlavB >= 0].b_jet_1_btagDeepFlavB.quantile(q=quantiles))

# Build signal quantiles dataframes
mask = sig.b_jet_1_btagDeepFlavB >= 0
sig_dfs = {}
sig_dfs['no_jet'] = {"weight":sig[~mask]['weight_central'].sum()/sig['weight_central'].sum(), 'df':sig[~mask]}
# Calculate bkg and data counts in each quantile
bkg_w, bkg_werr = {}, {}
data_w, data_werr = {}, {}
mask = bkg.b_jet_1_btagDeepFlavB >= 0
bkg_w['no_jet'] = bkg[~mask]['weight_central'].sum()/bkg['weight_central'].sum()
bkg_werr['no_jet'] = np.sqrt((bkg[~mask]['weight_central']**2).sum())/bkg['weight_central'].sum()
mask = data.b_jet_1_btagDeepFlavB >= 0
data_w['no_jet'] = data[~mask]['weight_central'].sum()/data['weight_central'].sum()
data_werr['no_jet'] = np.sqrt((data[~mask]['weight_central']**2).sum())/data['weight_central'].sum()
for i in range(n_quantiles):
    mask_btag = ( sig.b_jet_1_btagDeepFlavB >= btag_quantiles[i] )&( sig.b_jet_1_btagDeepFlavB < btag_quantiles[i+1] )
    sig_dfs['jet_q%g'%i] = {"weight":sig[mask_btag]['weight_central'].sum()/sig['weight_central'].sum(), 'df':sig[mask_btag]}
    # For background
    mask_btag = ( bkg.b_jet_1_btagDeepFlavB >= btag_quantiles[i] )&( bkg.b_jet_1_btagDeepFlavB < btag_quantiles[i+1] )
    bkg_w['jet_q%g'%i] = bkg[mask_btag]['weight_central'].sum()/bkg['weight_central'].sum()
    bkg_werr['jet_q%g'%i] = np.sqrt((bkg[mask_btag]['weight_central']**2).sum())/bkg['weight_central'].sum()
    mask_btag = ( data.b_jet_1_btagDeepFlavB >= btag_quantiles[i] )&( data.b_jet_1_btagDeepFlavB < btag_quantiles[i+1] )
    data_w['jet_q%g'%i] = data[mask_btag]['weight_central'].sum()/data['weight_central'].sum()
    data_werr['jet_q%g'%i] = np.sqrt((data[mask_btag]['weight_central']**2).sum())/data['weight_central'].sum()
    
# Calculate dropping no_jet bins
bkg_sumw_dropnojet, data_sumw_dropnojet = 0,0
bkg_w_dropnojet, data_w_dropnojet = {},{}
for k, v in bkg_w.items():
    if k != "no_jet":
        bkg_sumw_dropnojet += v
for k, v in bkg_w.items():
    if k != "no_jet":
        bkg_w_dropnojet[k] = v/bkg_sumw_dropnojet
for k, v in data_w.items():
    if k != "no_jet":
        data_sumw_dropnojet += v
for k, v in data_w.items():
    if k != "no_jet":
        data_w_dropnojet[k] = v/data_sumw_dropnojet

# Plotting steps

# Plot signal btag score distribution with vertical lines for quantiles
fig, ax = plt.subplots()
ax.hist(sig.b_jet_1_btagDeepFlavB[sig.b_jet_1_btagDeepFlavB >= 0], bins=50, range=(0,1), label="Signal", color='cornflowerblue', weights=sig[sig.b_jet_1_btagDeepFlavB >= 0]['weight_central'], density=True)
for q in btag_quantiles:
    ax.axvline(q, linestyle="--", zorder=9, linewidth=1)
ax.set_yscale("log")
ax.legend(loc="best")
ax.set_xlabel("Max b-tag score")
fig.savefig("plotting/hist_btag_score_signal.png")

# Plot bdt_score distributions for each quantile including no-jet: fractions in legend
fig, ax = plt.subplots()
for k,v in sig_dfs.items():
    if k == "no_jet": 
        name = "Zero jet"
        #ax.hist( v['df'].bdt_score, bins=20, range=(0.97,1), weights=v['df'].weight_central, density=True, label="%s (%.1f%%)"%(name,100*v['weight']), histtype='step', color='black')
        ax.hist( v['df'].bdt_score, bins=1000, range=(0,1), weights=v['df'].weight_central/v['df'].weight_central.sum(), label="%s (%.1f%%)"%(name,100*v['weight']), histtype='step', color='black')
    else: 
        name = "Quantile %s"%k.split("_q")[-1]
        #ax.hist( v['df'].bdt_score, bins=20, range=(0.97,1), weights=v['df'].weight_central, density=True, label="%s (%.1f%%)"%(name,100*v['weight']), histtype='step')
        ax.hist( v['df'].bdt_score, bins=1000, range=(0,1), weights=v['df'].weight_central/v['df'].weight_central.sum(), label="%s (%.1f%%)"%(name,100*v['weight']), histtype='step')

ax.set_xlim(0.97,1.0)
ax.legend(loc='best',fontsize=12)
ax.set_xlabel("BDT score")
for cut in [0.9891, 0.973610]:
    ax.axvline(cut, linestyle="--", zorder=9, linewidth=1)

fig.savefig("plotting/hist_bdt_score_signal_quantiles.png")


# Plot weighted sum of signal bdt_score histograms vs nominal distribution
fig,ax = plt.subplots()
n_bins = 1000
x_range = (0,1)
bc_wsum = np.zeros(n_bins)
for k, v in sig_dfs.items():
    #bc_wsum += v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central,density=True)[0]
    bc_wsum += v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central/v['df'].weight_central.sum())[0]

bc_wsum = bc_wsum/bc_wsum.sum()

#hist, xbins = np.histogram(sig.bdt_score, bins=n_bins, range=x_range, weights=sig.weight_central, density=True)
hist, xbins = np.histogram(sig.bdt_score, bins=n_bins, range=x_range, weights=sig.weight_central/sig.weight_central.sum())
hist = hist/hist.sum()

bin_centers = (xbins[:-1] + xbins[1:]) / 2
bin_width = xbins[1]-xbins[0]

ax.bar( bin_centers, hist, label="Signal", align="center", width=bin_width, color='cornflowerblue', alpha=0.5 )
ax.bar( bin_centers, bc_wsum, label="Signal (weighted quantile sum)", align="center", width=bin_width, color='None', edgecolor='red', alpha=0.5 )

#ax.step( xbins[:-1], hist, label="Signal", c='cornflowerblue')
#ax.step( xbins[:-1], bc_wsum, label="Signal (weighted sum)", c='orange')
ax.set_xlim(0.97,1)
ax.legend(loc='best')
ax.set_xlabel("BDT score")
for cut in [0.9891, 0.973610]:
  ax.axvline(cut, linestyle="--", zorder=9, linewidth=1)

fig.savefig("plotting/hist_bdt_score_signal_comparison.png")


# Plot bkg and data btag score distribution with quantiles
fig, ax = plt.subplots()
bin_edges = ax.hist(bkg.b_jet_1_btagDeepFlavB[bkg.b_jet_1_btagDeepFlavB >= 0], bins=50, range=(0,1), label="Background", color='grey', alpha=0.5, weights=bkg[bkg.b_jet_1_btagDeepFlavB >= 0]['weight_central'], density=True)
bin_centres = (bin_edges[1][:-1] + bin_edges[1][1:]) / 2
data_hist = np.histogram(data.b_jet_1_btagDeepFlavB[data.b_jet_1_btagDeepFlavB >= 0], bins=50, range=(0,1))[0]
data_err = bin_edges[0].sum()*(np.sqrt(data_hist)/data_hist.sum())
data_hist = bin_edges[0].sum()*(data_hist/data_hist.sum())
ax.errorbar(x=bin_centres, y=data_hist, yerr=data_err, fmt='o', capsize=2, color='black', label="Data")
for q in btag_quantiles:
    ax.axvline(q, linestyle="--", zorder=9, linewidth=1)

ax.set_yscale("log")
ax.legend(loc="best")
ax.set_xlabel("Max b-tag score")
fig.savefig("plotting/hist_btag_score_background.png")


# Plot ratio as fishbone with band for bkg MC and data error points
fig, ax = plt.subplots(figsize=(13,8))
bin_centers = np.linspace(0,n_quantiles,n_quantiles+1)+0.5
d = [ data_w['no_jet']/bkg_w['no_jet'] ]
de = [ data_werr['no_jet']/bkg_w['no_jet'] ]
b = [ bkg_w['no_jet']/bkg_w['no_jet'] ]
be = [ bkg_werr['no_jet']/bkg_w['no_jet'] ]

#Derive scale factors
for i in range(n_quantiles):
    key = "jet_q%g"%i
    d.append(data_w[key]/bkg_w[key])
    de.append(data_werr[key]/bkg_w[key])
    b.append(bkg_w[key]/bkg_w[key])
    be.append(bkg_werr[key]/bkg_w[key])

d = np.array(d)
de = np.array(de)
b = np.array(b)
be = np.array(be)
ax.bar( bin_centers, be, width=0.8, bottom=1, color='grey', alpha=0.5)
ax.bar( bin_centers, -1*be, width=0.8, bottom=1, color='grey', alpha=0.5, label='Bkg MC (stat unc)')
ax.errorbar( bin_centers, d, yerr=de, fmt='o', capsize=2, color='black', label='Data')
ax.set_ylim(0,2)
ax.axhline(1, linestyle='--', color='black')
ax.axvline(1, linestyle='--', color='black')
ax.legend(loc="best")
ax.set_xlabel("Max b-tag score quantiles")
ax.set_ylabel("Data / MC")
fig.savefig("plotting/ratio_btag_score_background.png")


# Plot reweighted fishbones for with and without no_jet
# Plot ratio as fishbone with band for bkg MC and data error points
fig, ax = plt.subplots(figsize=(13,8))
bin_centers = np.linspace(0,n_quantiles,n_quantiles+1)+0.5
d = [ data_w['no_jet']/bkg_w['no_jet'] ]
de = [ data_werr['no_jet']/bkg_w['no_jet'] ]
b = [ bkg_w['no_jet']/bkg_w['no_jet'] ]
be = [ bkg_werr['no_jet']/bkg_w['no_jet'] ]

for i in range(n_quantiles):
    key = "jet_q%g"%i
    d.append(data_w[key]/bkg_w[key])
    de.append(data_werr[key]/bkg_w[key])
    b.append(bkg_w[key]/bkg_w[key])
    be.append(bkg_werr[key]/bkg_w[key])

d = np.array(d)
de = np.array(de)
b = np.array(b)
be = np.array(be)
ax.bar( bin_centers, be, width=0.8, bottom=1, color='grey', alpha=0.5)
ax.bar( bin_centers, -1*be, width=0.8, bottom=1, color='grey', alpha=0.5, label='Bkg MC (stat unc)')
ax.errorbar( bin_centers, d, yerr=de, fmt='o', capsize=2, color='black', label='Data')
ax.errorbar( bin_centers, b, yerr=de, fmt='o', capsize=2, color='red', label='Reweighting')
ax.set_ylim(0,2)
ax.axhline(1, linestyle='--', color='black')
ax.axvline(1, linestyle='--', color='black')
ax.legend(loc="best")
ax.set_xlabel("Max b-tag score quantiles")
ax.set_ylabel("Data / MC")
fig.savefig("plotting/ratio_btag_score_background_rwgt_illustration.png")



# Plot reweighted bdt score distribution for signal
fig,ax = plt.subplots()
n_bins = 1000
x_range = (0,1)
bc_wsum, bc_wsum_morphed = np.zeros(n_bins), np.zeros(n_bins)
d_itr = 0
for k, v in sig_dfs.items():
    #bc_wsum += v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central,density=True)[0]
    bc_wsum += v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central/v['df'].weight_central.sum())[0]
    #bc_wsum_morphed += d[d_itr]*v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central,density=True)[0]
    bc_wsum_morphed += d[d_itr]*v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central/v['df'].weight_central.sum())[0]
    d_itr += 1

bc_wsum = bc_wsum/bc_wsum.sum()
bc_wsum_morphed = bc_wsum_morphed/bc_wsum_morphed.sum()

_, xbins = np.histogram([], bins=n_bins, range=x_range)
bin_centers = (xbins[:-1] + xbins[1:]) / 2
bin_width = xbins[1]-xbins[0]

ax.bar( bin_centers, bc_wsum, label="Signal (nominal)", align="center", width=bin_width, color='cornflowerblue', alpha=0.5 )
ax.errorbar( bin_centers, bc_wsum_morphed, label="Signal (reweighted)", color='red', fmt='o', alpha=0.5 )
#ax.step( xbins[:-1], bc_wsum, label="Signal (nominal)", c='cornflowerblue')
#ax.step( xbins[:-1], bc_wsum_morphed, label="Signal (reweighted)", c='red')
ax.set_xlim(0.97,1)
ax.legend(loc='best')
ax.set_xlabel("BDT score")
for cut in [0.9891, 0.973610]:
  ax.axvline(cut, linestyle="--", zorder=9, linewidth=1)

fig.savefig("plotting/hist_bdt_score_signal_reweighted.png")


# Calculate variation in signal efficiency for SR0, SR1 and outside
SR0_y, SR0_y_rwgt = 0, 0
SR1_y, SR1_y_rwgt = 0, 0
tot_y, tot_y_rwgt = 0, 0
i = 0
for k, v in sig_dfs.items():
    df = v['df']
    #SR0
    mask_SR0 = df.bdt_score >= 0.9891
    SR0_y += df[mask_SR0].weight_central.sum()
    SR0_y_rwgt += d[i]*df[mask_SR0].weight_central.sum()
    #SR1
    mask_SR1 = (df.bdt_score >= 0.973610)&(df.bdt_score < 0.9891)
    SR1_y += df[mask_SR1].weight_central.sum()
    SR1_y_rwgt += d[i]*df[mask_SR1].weight_central.sum()
    # Total
    tot_y += df.weight_central.sum()
    tot_y_rwgt += d[i]*df.weight_central.sum()
    i += 1

print("[Nominal]    Efficiency:  SR0 = %.4f,    SR1 = %.4f"%(SR0_y/tot_y,SR1_y/tot_y))
print("[Reweighted] Efficiency:  SR0 = %.4f,    SR1 = %.4f"%(SR0_y_rwgt/tot_y_rwgt,SR1_y_rwgt/tot_y_rwgt))


# Plot reweighted fishbones without no_jet
# Plot ratio as fishbone with band for bkg MC and data error points
fig, ax = plt.subplots(figsize=(13,8))
bin_centers = np.linspace(0,n_quantiles-1,n_quantiles)+0.5
d = []
de = []
b = []
be = []

# Derive scale factors without zero-jet
for i in range(n_quantiles):
    key = "jet_q%g"%i
    d.append(data_w_dropnojet[key]/bkg_w_dropnojet[key])
    de.append(data_werr[key]/bkg_w_dropnojet[key])
    b.append(bkg_w_dropnojet[key]/bkg_w_dropnojet[key])
    be.append(bkg_werr[key]/bkg_w_dropnojet[key])

d = np.array(d)
de = np.array(de)
b = np.array(b)
be = np.array(be)
ax.bar( bin_centers, be, width=0.8, bottom=1, color='grey', alpha=0.5)
ax.bar( bin_centers, -1*be, width=0.8, bottom=1, color='grey', alpha=0.5, label='Bkg MC (stat unc)')
ax.errorbar( bin_centers, d, yerr=de, fmt='o', capsize=2, color='black', label='Data')
ax.errorbar( bin_centers, b, yerr=de, fmt='o', capsize=2, color='red', label='Reweighting')
ax.set_ylim(0,2)
ax.axhline(1, linestyle='--', color='black')
ax.legend(loc="best")
ax.set_xlabel("Max b-tag score quantiles")
ax.set_ylabel("Data / MC")
fig.savefig("plotting/ratio_btag_score_background_rwgt_illustration_dropnojet.png")

# Plot reweighted bdt score distribution for signal
fig,ax = plt.subplots()
n_bins = 1000
x_range = (0,1)
bc_wsum, bc_wsum_morphed = np.zeros(n_bins), np.zeros(n_bins)
d_itr = 0
for k, v in sig_dfs.items():
    if k != "no_jet":
        bc_wsum += v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central/v['df'].weight_central.sum())[0]
        bc_wsum_morphed += d[d_itr]*v['weight']*np.histogram(v['df'].bdt_score, bins=n_bins, range=x_range, weights=v['df'].weight_central/v['df'].weight_central.sum())[0]
        d_itr += 1

bc_wsum = bc_wsum/bc_wsum.sum()
bc_wsum_morphed = bc_wsum_morphed/bc_wsum_morphed.sum()

# Add no jet with correct weighting
w_nojet = sig_dfs['no_jet']['weight']

bc_wsum = w_nojet*np.histogram(sig_dfs['no_jet']['df'].bdt_score, bins=n_bins, range=x_range, weights=sig_dfs['no_jet']['df'].weight_central/sig_dfs['no_jet']['df'].weight_central.sum())[0] + (1-w_nojet)*bc_wsum

bc_wsum_morphed = w_nojet*np.histogram(sig_dfs['no_jet']['df'].bdt_score, bins=n_bins, range=x_range, weights=sig_dfs['no_jet']['df'].weight_central/sig_dfs['no_jet']['df'].weight_central.sum())[0] + (1-w_nojet)*bc_wsum_morphed

ns = np.histogram([], bins=n_bins, range=x_range)
bin_centers = (xbins[:-1] + xbins[1:]) / 2
bin_width = xbins[1]-xbins[0]

ax.bar( bin_centers, bc_wsum, label="Signal (nominal)", align="center", width=bin_width, color='cornflowerblue', alpha=0.5 )
ax.errorbar( bin_centers, bc_wsum_morphed, label="Signal (reweighted, fix 0-jet)", color='red', fmt='o', alpha=0.5 )
#ax.step( xbins[:-1], bc_wsum, label="Signal (nominal)", c='cornflowerblue')
#ax.step( xbins[:-1], bc_wsum_morphed, label="Signal (reweighted)", c='red')
ax.set_xlim(0.97,1)
ax.legend(loc='best')
ax.set_xlabel("BDT score")
for cut in [0.9891, 0.973610]:
  ax.axvline(cut, linestyle="--", zorder=9, linewidth=1)

fig.savefig("plotting/hist_bdt_score_signal_reweighted_dropnojet.png")

# Calculate variation in signal efficiency for SR0, SR1 and outside
SR0_y, SR0_y_rwgt = 0, 0
SR1_y, SR1_y_rwgt = 0, 0
tot_y, tot_y_rwgt = 0, 0
i = 0
for k, v in sig_dfs.items():
    df = v['df']
    #SR0
    mask_SR0 = df.bdt_score >= 0.9891
    SR0_y += df[mask_SR0].weight_central.sum()
    if k != 'no_jet':
        SR0_y_rwgt += d[i]*df[mask_SR0].weight_central.sum()
    else:
        SR0_y_rwgt += df[mask_SR0].weight_central.sum()
    #SR1
    mask_SR1 = (df.bdt_score >= 0.973610)&(df.bdt_score < 0.9891)
    SR1_y += df[mask_SR1].weight_central.sum()
    if k != 'no_jet':
        SR1_y_rwgt += d[i]*df[mask_SR1].weight_central.sum()
    else:
        SR1_y_rwgt += df[mask_SR1].weight_central.sum()
    # Total
    tot_y += df.weight_central.sum()
    if k != 'no_jet':
        tot_y_rwgt += d[i]*df.weight_central.sum()
        i += 1
    else:
        tot_y_rwgt += df.weight_central.sum()

print("\n\n\n\n\n[Nominal]               Efficiency:  SR0 = %.4f,    SR1 = %.4f"%(SR0_y/tot_y,SR1_y/tot_y))
print("[Reweighted, fix 0-jet] Efficiency:  SR0 = %.4f,    SR1 = %.4f"%(SR0_y_rwgt/tot_y_rwgt,SR1_y_rwgt/tot_y_rwgt))





