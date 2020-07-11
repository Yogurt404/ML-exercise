import numpy as np
from dict_map import ClassMap

classmap = ClassMap()

pred = {}
with open('pin/results/test.txt', 'r') as f:
    ls = f.readlines()

for l in ls:
    a = l.split(' ')
    assert len(a) == 2
    idx = classmap.get_idx(a[1][:-1])
    pred.update({a[0]: idx})

manual = {}
with open('pin/manual.txt', 'r') as f:
    ls = f.readlines()
for l in ls:
    a = l.split(' ')
    assert len(a) == 2
    idx = classmap.get_idx(a[1][:-1])
    manual.update({a[0]: idx})

n = 0
for key in manual:
    if manual[key] == pred[key]:
        n += 1

print(n/len(manual))