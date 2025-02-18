import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys
import common
import numpy as np
import scipy.optimize as spo

from sklearn.neighbors import KernelDensity
from scipy.stats import powerlaw

pl = lambda x, a, b, c: powerlaw.pdf(x, a, loc=b, scale=c)

def fitFunction(x, a, b, c):
  return a/(x-b) + c

def generateToys(f, a, b, n):
  poss_toys = np.random.rand(n*2)*(b-a) + a
  p = f(poss_toys)
  p /= sum(p)
  return np.random.choice(poss_toys, size=n, p=p)

df1_path = sys.argv[1]
df2_path = sys.argv[2]

scores_1 = list(filter(lambda x: "score" in x, common.getColumns(df1_path)))
scores_2 = list(filter(lambda x: "score" in x, common.getColumns(df2_path)))
scores = list(set(scores_1).intersection(scores_2))
print(scores)

for score in scores:
  df1 = pd.read_parquet(df1_path, columns=["weight", "process_id", score])
  df1 = df1[df1.process_id==0]
  df2 = pd.read_parquet(df2_path, columns=["weight", "process_id", score])
  df2 = df2[df2.process_id==0]

  df1 = df1[df1[score] > 0.5]
  df1.loc[:, "weight"] /= df1.weight.sum()

  r = (0.5, 1.0)
  nbins = 50
  sumw, bins = np.histogram(df1[score], range=r, bins=nbins, weights=df1.weight)
  sumw2, bins = np.histogram(df1[score], range=r, bins=nbins, weights=df1.weight**2)
  error = np.sqrt(sumw2)

  sumw *= nbins / (r[1]-r[0])
  error *= nbins / (r[1]-r[0])
  #print(sumw)

  #n, bins, patches = plt.hist(df1[score], range=(0.5,0.99), bins=50, histtype="step", density=True)
  #plt.hist(df2[score], range=(0.5,0.99), bins=50, histtype="step", density=True)

  x = (bins[:-1]+bins[1:])/2

  plt.errorbar(x, sumw, error, fmt='x')
  plt.ylim(bottom=0)

  popt, pcov = spo.curve_fit(fitFunction, x, sumw, p0=[1, 1.5, 1], bounds=[[0, 1+1e-3, 0], [100, 10, 10]])
  plt.plot(x, fitFunction(x, *popt))

  #toys = generateToys(lambda x: fitFunction(x, *popt), 0.5, 1.0, 100000)
  #plt.hist(toys, range=r, bins=nbins, density=True, histtype="step")

  #popt, pcov = spo.curve_fit(pl, x, sumw, p0=[1.5, 1.5, -0.5], bounds=[[1, 1+1e-2, -1], [2, 2, 0]])
  #popt = [1.5, 1.5, -0.5]
  #plt.plot(x, pl(x, *popt))

  df1_score = df1[score]
  df1_score = df1_score[df1_score>0.5]
  #df1_score = pd.concat([df1_score, 2-df1_score])
  kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(df1_score.to_numpy().reshape(-1, 1))
  
  x = np.linspace(0.5, 1, 100)
  
  plt.plot(x, np.exp(kde.score_samples(x.reshape(-1, 1))))

  print(score, popt)

  plt.savefig("data_photon_id_compare/%s.png"%score)
  plt.clf()
