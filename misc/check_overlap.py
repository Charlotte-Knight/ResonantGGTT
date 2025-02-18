import sys
import os
import uproot
import numpy as np

events = []
for i, d in enumerate(sys.argv[1:]):
  events.append([])
  files = os.listdir(d)
  files = list(filter(lambda x: "mx600my90" in x, files))
   
  for name in files:
    cat_no = int(name.split("cat")[1][0])
    # if cat_no >= len(files) - 3:
    #   continue
    if cat_no != 0:
      continue
    path = os.path.join(d, name)
    f = uproot.open(path)
    #events[i].append(f[f.keys()[0]]["event"].array())
    events[i].append(f[f.keys()[0]]["CMS_hgg_mass"].array())

  events[i] = np.concatenate(events[i])

  assert len(np.unique(events[i])) == len(events[i])

events = [set(event_set) for event_set in events]

overlap = set.intersection(*events)
print(overlap)

fraction_intersection = [len(overlap)/len(event_set) for event_set in events]
print(fraction_intersection)