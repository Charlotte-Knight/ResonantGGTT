import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generateToys(ntoys, events_per_toy=1000):
  size = ntoys * events_per_toy  

  #toys = sps.expon.rvs(size=size)
  toys = []

  toys = np.concatenate([toys, sps.norm.rvs(loc=0.5, scale=0.01, size=1000)])
  np.random.shuffle(toys)
  return np.split(toys, ntoys)

def plotToy(toy, fit_f=None):
  hist, edges = np.histogram(toy, range=(0, 1), bins=50)
  centers = (edges[:-1] + edges[1:]) / 2
  
  s = hist != 0
  hist = hist[s]
  centers = centers[s]

  plt.scatter(centers, hist)
  plt.plot(centers, fit_f(centers))
  plt.xlim(0, 1)

  plt.savefig("toy.png")

def fitToy(toy, m):
  def sig(x, m):
    return sps.norm.pdf(x, loc=m, scale=0.1)

  def bkg(x):
    return sps.expon.pdf(x)

  def spb(x, m, f):
    return f*sig(x, m) + (1-f)*bkg(x)

  def NLL(f, x, m):
    NLL = -np.log(spb(x, m, f)).sum()
    return NLL

  res = spo.minimize(NLL, [0.1], args=(toy, m), bounds=[(0, 1)])
  print(res.x)
    
  plotToy(toy, lambda x: spb(x, m, res.x[0]))
  
  #return {"nsig": nsig, "nbkg": nbkg}
  
toys = generateToys(10)
#plotToy(toys[0])
fitToy(toys[0], 125)
