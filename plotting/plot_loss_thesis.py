import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
import pickle
import sys
sys.path.append("training")
import pandas as pd
import numpy as np
    
def plotLossAxis(model, ax, idx=None):
  if idx:
    train_loss = pd.Series(model.train_loss[:,idx])
    val_loss = pd.Series(model.validation_loss[:,idx])
  else:
    train_loss = pd.Series(model.train_loss.sum(axis=1))
    val_loss = pd.Series(model.validation_loss.sum(axis=1))
  
  window_size = 10
  smoothed_train_loss = train_loss.rolling(window=window_size, min_periods=1, center=True).mean()
  smoothed_val_loss = val_loss.rolling(window=window_size, min_periods=1, center=True).mean()
  
  x = np.arange(len(train_loss)) + 1
  
  ax.plot(x, smoothed_train_loss, label="Training", color="tab:blue")
  ax.plot(x, smoothed_val_loss, label="Validation", color="tab:orange")
  ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
  ax.plot(x, train_loss, alpha=0.5, color="tab:blue")
  ax.plot(x, val_loss, alpha=0.5, color="tab:orange")

def plotLoss(indir):
  with open(f"{indir}/model.pkl", "rb") as f:
    model = pickle.load(f)["classifier"]
  
  plt.figure(figsize=(10, 8))
  plotLossAxis(model, plt.gca())

  plt.xlabel("Epoch")
  plt.ylabel("Weighted BCE Loss")
  plt.legend()
  plt.tight_layout()
  plt.savefig(f"{indir}/loss.pdf")
  plt.clf()

def plotSplitLoss(indir):
  mXs = [250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]
  
  with open(f"{indir}/model.pkl", "rb") as f:
    model = pickle.load(f)["classifier"]
        
  fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 13))
  plt.subplots_adjust(hspace=0.05, bottom=0.1, top=0.95)
  for i, m in enumerate([260, 500, 1000]):
    idx = mXs.index(m)
    plotLossAxis(model, axs[i], idx)
    axs[i].text(0.95, 0.8, r"$m_X = %d\,GeV$"%m, transform=axs[i].transAxes, horizontalalignment="right", fontsize=19)
  plotLossAxis(model, axs[3])
  axs[3].text(0.95, 0.8, "All masses", transform=axs[3].transAxes, horizontalalignment="right", fontsize=19)
  
  axs[3].set_xlabel("Epoch")
  axs[0].set_ylabel("Weighted BCE Loss")
  axs[0].legend(loc="upper center")
  #plt.tight_layout()
  plt.savefig(f"{indir}/loss_split.pdf")
  plt.clf()

plotLoss("Outputs/Graviton")
plotLoss("Outputs/Radion")
plotLoss("Outputs/Y_tautau")
plotLoss("Outputs/Y_gg_Low_Mass")
plotLoss("Outputs/Y_gg_High_Mass_no_dilep_leadpho_mass")

plotSplitLoss("Outputs/Graviton")