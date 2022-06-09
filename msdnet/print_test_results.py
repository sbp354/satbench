import os, glob
import json

results = {}

for file in glob.glob("*.out"):
	lines = open(file, 'r').readlines()
	
	modeval = None
	for l in lines:
		if 'NOISE' in l or 'BLUR' in l:
			modeval = l.split(' ')[-1][:4].strip()
			break

	if modeval == None:
		if "g." in file:
			modeval = 'gray'
		elif "c." in file:
			modeval = 'color'

	results[modeval] = {}

	flops = map(float, lines[-2][8:-2].strip().split(', '))
	accs = map(float, lines[-1][7:-2].strip().split(', '))

	results[modeval]['flops'] = flops
	results[modeval]['accs'] = accs

# print (json.dumps(results, indent=2, default=str))
print results
