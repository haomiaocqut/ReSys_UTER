import json

data = []
with open('../inputdata/meta_Baby.json') as f:
    for line in f:
        linedict = json.dumps(eval(line))
        data.append(json.loads(linedict))
    f.close()

itemdict = {}
unum = 1

for ui in data:
    if (ui["asin"] not in itemdict):
        itemdict[ui["asin"]] = unum
    else:
        print(itemdict)
        itemdict[ui["asin"]] +=1

print (sorted(itemdict.items(), reverse= False, key=lambda d: d[1])[0:1000])


