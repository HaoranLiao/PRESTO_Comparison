import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from frb_L2_L3 import L1_event as ev

def plot(ax, data, basenm, timelim_low, timelim_high):
	dm_mean = np.mean(data['dm_best'])
	dm_std = np.std(data['dm_best'])
	dm_stderr = dm_std/len(data['dm_best'])
	print(dm_mean)
	print(dm_stderr)
	#print(data['dm_best'])
	#fig = plt.figure(figsize=(13, 4))
	for i in range(len(data)):
		if data['time'][i]<=timelim_high and data['time'][i]>=timelim_low:
			#plt.scatter(data['time'][i], data['dm_best'][i], s=data['snr'][i]/5
			asym_err = [data['dm_best'][i]-data['dm_min'][i],data['dm_max'][i]-data['dm_best'][i]]
			asym_err_2 = [data['dm_best'][i]-data['dm_best_min'][i],data['dm_best_max'][i]-data['dm_best'][i]]
			#print(asym_err)
			#plt.errorbar(data['time'][i], data['dm_best'][i], yerr=[asym_err], marker='o', markersize=2, color='black', lw=0.5, capsize=0, capthick=0)
			#plt.errorbar(data['time'][i], data['dm_best'][i], yerr=[asym_err_2], marker='o', markersize=2, color='black', lw=0.5, capsize=1.5, capthick=0.5)
			ax.errorbar(data['time'][i], data['dm_best'][i], yerr=[asym_err], marker='o', markersize=2, color='black', lw=0.5, capsize=0, capthick=0)
			ax.errorbar(data['time'][i], data['dm_best'][i], yerr=[asym_err_2], marker='o', markersize=2, color='black', lw=0.5, capsize=1.5, capthick=0.5)
			#plt.errorbar(data['time'][i], data['dm_best'][i], marker='o', markersize=data['snr'][i]/5, yerr=[data['dm_min'][i],data['dm_max'][i]])
	#ax = plt.gca()
	ax.set_xlim([timelim_low, timelim_high])
	ax.set_ylim([0,50])
	ax.set_ylabel('DM ($pc$ $cm^{-3}$)')
	#plt.ylabel('DM ($pc$ $cm^{-3}$)')
	#plt.yticks(np.arange(0, 60, 10))
	x = np.linspace(timelim_low, timelim_high)
	#plt.plot(x, [dm_mean]*len(x), linestyle='--', lw=0.5, color='black')
	#plt.plot(x, [26.76]*len(x), linestyle='-', lw=0.5, color='red')
	ax.plot(x, [dm_mean]*len(x), linestyle='--', lw=0.5, color='black')
	ax.plot(x, [26.76]*len(x), linestyle='-', lw=0.5, color='red')
	#fig.savefig('%s_pulses.png'%basenm)

	
	#plt.show()

def time_histo(array):
	import matplotlib.pyplot as plt
	array.sort()
	print(len(array))
	diff = []
	dup_count = 0
	for i in range(len(array)):
		if i>0:
			diff.append(array[i]-array[i-1])
			if array[i]-array[i-1]<0.3:
				dup_count += 1
	print(dup_count)
	n, bins, patches = plt.hist(diff, 40, range=[0,8],facecolor='black', align='mid')
	plt.title('Neigbhouring Detections Time Separation')
	plt.xlabel('Time (s)')
	plt.ylabel('Count')
	plt.xticks(np.arange(0,8,0.7))
	axes = plt.gca()
	axes.set_ylim([0,200])
	plt.show()

def shape(mask):
	maskfrac = []
	for i in range(len(mask)):
		x = mask[i][mask[i]<=0.3]
		maskfrac.append(x.size/float(mask[i].size))
	bar_info = np.zeros(len(mask), dtype=[('left', np.float32), ('height', np.float32), ('width', np.float32)])
	tsamp = 0.00131072
	for i in range(len(mask)):
		bar_info['height'][i] = maskfrac[i]
		bar_info['width'][i] = len(mask[i])*tsamp
		if i>0:
			bar_info['left'][i] = bar_info['left'][i-1]+bar_info['width'][i-1]
		else:
			bar_info['left'][0] = 0
	
	for i in range(len(mask)):	
		print('Left: %f   Width: %f   Height: %f'%(bar_info['left'][i],bar_info['width'][i],bar_info['height'][i]))
	
	return bar_info

def plotbar(ax, maskfrac_bar):
	ax.bar(maskfrac_bar['left'], maskfrac_bar['height'], maskfrac_bar['width'], color='b', alpha=0.5, align='center')
	ax.set_ylabel('Masking Fraction')
	#plt.ylabel('Masking Fraction')
	#plt.show()

def main():
	infilenm = sys.argv[1]
	basenm = infilenm.split('.')[0]
	data = np.load(infilenm)
	#print(data)
	x = np.zeros(len(data), dtype=[('beam', np.int16), ('itree', np.int16), ('snr', np.float32), ('time', np.float32), ('dm_min', np.float32), ('dm_max', np.float32),\
								   ('dm_best_min', np.float32), ('dm_best_max', np.float32), ('dm_best', np.float32), ('grade', np.float32)])
	for i in range(len(data)):
	#for i in range(2):
		beam = data[i]['beam']
		itree = data[i]['itree']
		snr = data[i]['snr']
		time = data[i]['time'].astype(float)/1e6
		dm_min = ev.dms_for_snr_vs_dm(data[i])[0]
		if itree==0:
			dm_max_ind = 16
		elif itree==1:
			dm_max_ind = 8
		elif itree==2:
			dm_max_ind = 4
		dm_max = ev.dms_for_snr_vs_dm(data[i])[dm_max_ind]+ev.tree_ddm(itree)
		dm_best_min = data[i]['dm']
		dm_best_max = data[i]['dm']+ev.tree_ddm(itree)
		dm_best = dm_best_min+(dm_best_max-dm_best_min)/2
		#snr_vs_dm = data[i]['snr_vs_dm']
		grade = data[i]['grade']
		#snr_vs_itree = data[i]['snr_vs_itree']
		x[i] = (beam, itree, snr, time, dm_min, dm_max, dm_best_min, dm_best_max, dm_best, grade)
	x = x[np.argsort(x['time'])]
	#print(x)
	#print(len(data))
	#print(len(x))

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	#font = {'family' : 'normal',\
	#		#'weight' : 'bold',\
	#		'size'   : 16}

	#matplotlib.rc('font', **font)

	inmasknm = sys.argv[2]
	mask = np.load(inmasknm)
	maskfrac_bar = shape(mask)
		
	plot(ax1, x, basenm, 0, 410)
	plotbar(ax2, maskfrac_bar)

	#plot(x, basenm, 0, 200)
	#plot(x, basenm, 200, 410)
	#time_histo(x['time'])

	ax1.set_title('Pulses Grouped by L1')
	ax1.set_xlabel('Arrival Time ($s$)')
	plt.show()

if __name__ == "__main__":
    main()
