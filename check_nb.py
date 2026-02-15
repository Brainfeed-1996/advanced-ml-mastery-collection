import json
nb = json.load(open('02-Classification/Random-Forest-Finance.ipynb'))
outs = sum(1 for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs'))
print('outputs:', outs)
