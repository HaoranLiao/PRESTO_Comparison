import numpy as np
import sys
import matplotlib.pyplot as plt

def readsps(infilenm):
	basenm = infilenm.split('.')[0]
	with open(infilenm) as f:
		lines = f.readlines()
	num_total = len(lines)
	#print(lines[288])
	print(num_total)
	x = np.zeros(num_total-1, dtype=[('dm',np.float32),('sigma',np.float32),('time',np.float32),('sample',np.int16),('downfact',np.int16)])
	for i in range(num_total-1):
		curr_line = lines[i+1]
		entry = curr_line.split()
		dm, sigma, time, sample, downfact = round(float(entry[0].strip()),2), round(float(entry[1].strip()),2), round(float(entry[2].strip()),4), \
											entry[3].strip(), entry[4].strip()
		x[i] = (dm, sigma, time, sample, downfact)
	x = x[np.argsort(x['time'])]
	return x

def readnpy(infilenm):
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
		#dm_min = ev.dms_for_snr_vs_dm(data[i])[0]
		dm_min = 0
		if itree==0:
			dm_max_ind = 16
		elif itree==1:
			dm_max_ind = 8
		elif itree==2:
			dm_max_ind = 4
		#dm_max = ev.dms_for_snr_vs_dm(data[i])[dm_max_ind]+ev.tree_ddm(itree)
		dm_max = 0
		dm_best_min = data[i]['dm']
		#dm_best_max = data[i]['dm']+ev.tree_ddm(itree)
		#dm_best = dm_best_min+(dm_best_max-dm_best_min)/2
		#snr_vs_dm = data[i]['snr_vs_dm']
		dm_best_max = 0
		dm_best = 0
		grade = data[i]['grade']
		#snr_vs_itree = data[i]['snr_vs_itree']
		x[i] = (beam, itree, snr, time, dm_min, dm_max, dm_best_min, dm_best_max, dm_best, grade)
	x = x[np.argsort(x['time'])]
	return x
	
def fill_t_series_npy(p, series):
	#series = series_full['t']
	print('length of the series: %d'%len(series))
	series_nodup = [series[0]]
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			if diff>p/2:
				series_nodup.append(series[i])
	series = series_nodup
	print('length of ther series nodup: %d'%len(series))

	series_filled = []
	#series_std = []
	#series_std.append(series[0])
	series_filled.append(series[0])
	count = 1
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			num_p = int(round(diff/p))
			#print('num_p: %d'%num_p)
			series_filled.extend([0]*(num_p-1))
			series_filled.append(series[i])
			#print(series_filled)
			count += num_p
			#print(count)
	print('number of standard pulses during series time: %d'%count)
	num_zero_to_add_begin = int(round((series[0])/p))
	num_zero_to_add_end = 571-count-num_zero_to_add_begin+1
	#print(num_zero_to_add_end)
	series_filled = [0]*num_zero_to_add_begin+series_filled+[0]*num_zero_to_add_end
	series_std = np.arange(0,series[0]+(count)*p,p)
	#print(series_std)
	#print(series_filled)
	print('last element: %.1f'%series_std[-1])
	print('last element: %.1f'%series_filled[-1])
	print('Length of filled series: %d'%len(series_filled))
	print('Length of standard series: %d'%len(series_std))
	return series_filled, count

def fill_t_series_sps(p, series):
	#series = series_full['t']
	print('length of the series: %d'%len(series))
	series_nodup = [series[0]]
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			if diff>p/2:
				series_nodup.append(series[i])
	series = series_nodup
	print('length of ther series nodup: %d'%len(series))

	series_filled = []
	#series_std = []
	#series_std.append(series[0])
	series_filled.append(series[0]-0.2)
	count = 1
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			num_p = int(round(diff/p))
			#print('num_p: %d'%num_p)
			series_filled.extend([0]*(num_p-1))
			series_filled.append(series[i]-0.2)
			#print(series_filled)
			count += num_p
			#print(count)
	print('number of standard pulses during series time: %d'%count)
	num_zero_to_add_begin = int(round((series[0])/p))
	num_zero_to_add_end = 571-count-num_zero_to_add_begin+1
	#print(num_zero_to_add_end)
	series_filled = [0]*num_zero_to_add_begin+series_filled+[0]*num_zero_to_add_end
	series_std = np.arange(0,series[0]+(count-1)*p,p)
	#print(series_std)
	#print(series_filled)
	print('last element: %.1f'%series_std[-1])
	print('last element: %.1f'%series_filled[-1])
	print('Length of filled series: %d'%len(series_filled))
	print('Length of standard series: %d'%len(series_std))
	return series_filled, count

def time_histo(array):
    
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
    plt.title('PRESTO Neigbhouring Detections Time Separation')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.xticks(np.arange(0,8,0.7))
    axes = plt.gca()
    axes.set_ylim([0,200])
    plt.show()

def remove_npy_dup(data, p):
	data = data
	print(len(data['time']))
	ind = []
	for i in range(310):
		if i>0:
			diff = data['time'][i]-data['time'][i-1]
			#print(diff)
			if diff<0.5*p:
				ind.append(i)
	data = np.delete(data, ind)
	print(len(data))
	return data


def snr_plot(snr, sps=None):
	n, bins, _ = plt.hist(snr, bins=np.arange(0, 50, 2), range=[0,50], histtype='step', color='black', align='mid')
	mid = 0.5*(bins[1:] + bins[:-1])
	plt.errorbar(mid, n, yerr=np.sqrt(n), fmt=None, c='black', capsize=2, lw=1)
	plt.xlabel('Signal-to-noise Ration')
	if sps is None:
		plt.title('Detection SNR by L1')
	else:
		plt.title('Detection SNR by Single Pulse Search (DM %s)'%sps)
	plt.ylabel('Count')
	ax = plt.gca()
	plt.savefig('snr_sps_DM%s.png'%sps)
	plt.show()

def snr_plot2(snr1, snr2):
	n1, bins1, _ = plt.hist(snr1, bins=np.arange(0, 50, 2), range=[0,50], histtype='step', color='black', align='mid')
	n2, bins2, _ = plt.hist(snr2, bins=np.arange(0, 50, 2), range=[0,50], histtype='step', color='red', align='mid')
	mid1 = 0.5*(bins1[1:] + bins1[:-1])
	#plt.errorbar(mid1, n1, yerr=np.sqrt(n1), fmt=None, c='black', capsize=2, lw=1)
	mid2 = 0.5*(bins2[1:] + bins2[:-1])
	#plt.errorbar(mid2, n2, yerr=np.sqrt(n2), fmt=None, c='red', capsize=2, lw=1)
	plt.xlabel('Signal-to-noise Ration')
	plt.title('Distribution of Detection SNR By Single Pulse Search')
	plt.ylabel('Count')
	ax = plt.gca()
	plt.show()

def main():

	sps_result = readsps(sys.argv[1])
	DM_value = sys.argv[1].split('.')[0].split('DM')[1]+'.'+sys.argv[1].split('.')[1]
	#print(DM_value)
	#npy_result = readnpy(sys.argv[1])
	#npy_result2 = readnpy(sys.argv[2])
	#print(npy_result['snr'])

	#npy_result = remove_npy_dup(npy_result, 0.714)
	sps_result = remove_npy_dup(sps_result, 0.714)
	#print(npy_result)
	#print(len(npy_result))
	#snr_plot2(npy_result['snr'], npy_result2['snr'])
	snr_plot(sps_result['sigma'], sps=DM_value)
	




if __name__=='__main__':
	main()
