import os
import json
ls = os.listdir('save/SVHN-json')
print(ls)
accs = []
for l in ls:
    with open('save/SVHN-json/'+l, "r", encoding='utf-8') as r:
        dic = json.load(r)
        acc = []
        for d in dic:
            acc.append(d[2])
    accs.append(acc)
print(accs)