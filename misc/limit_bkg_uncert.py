from optimisation.limit import calculateExpectedLimit
import numpy as np
import pandas as pd

s = np.array([1, 5, 10, 100])
#s = np.array([1])
b = np.array([0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
b_err = np.sqrt(b)*0.99
b_err[b<1] = 0.5*b[b<1]

results = []

for si in s:
  for i, bi in enumerate(b):
    b_erri = b_err[i]
    results.append([si, bi, b_erri, calculateExpectedLimit(si, bi, 0), calculateExpectedLimit(si, bi, b_erri), calculateExpectedLimit(si, bi-b_erri, 0), calculateExpectedLimit(si, bi+b_erri, 0), (calculateExpectedLimit(si, bi-b_erri, 0)+calculateExpectedLimit(si, bi+b_erri, 0))/2])

df = pd.DataFrame(results, columns=["s", "b", "b_err", "no uncert", "uncert", "-1", "+1", "avg"])
print(df)