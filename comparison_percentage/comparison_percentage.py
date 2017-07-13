import numpy as np
import sys

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
    plt.title('PRESTO Neigbhouring Detections Time Separation')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.xticks(np.arange(0,8,0.7))
    axes = plt.gca()
    axes.set_ylim([0,200])
    plt.show()


def main():

	sps_result = readsps(sys.argv[1])
	npy_result = readnpy(sys.argv[2])
	npy2_result = readnpy(sys.argv[3])

	time_histo(sps_result['time'])

	series_filled_sps, count_sps = fill_t_series_sps(0.71446, sps_result['time'])

	series_filled_npy, count_npy = fill_t_series_npy(0.71446, npy_result['time'])
	series_filled_npy2, count_npy2 = fill_t_series_npy(0.71446, npy2_result['time'])


	print('length of the series_filled_sps: %d'%len(series_filled_sps))
	print('length of the series_filled_npy: %d'%len(series_filled_npy))
	print('length of the series_filled_npy2: %d'%len(series_filled_npy2))

	count_both = 0
	count_bothmiss = 0
	count_npy_only = 0
	count_sps_only = 0
	for i in range(len(series_filled_sps)):
		#print('%.1f    %.1f'%(series_filled_sps[i], series_filled_npy[i]))
		#print('%.1f    %.1f'%(series_filled_sps[i], series_filled_npy[i]))
		if series_filled_npy[i]!=0 and series_filled_sps[i]!=0:
			count_both += 1
		elif series_filled_npy[i]!=0 and series_filled_sps[i]==0:
			count_npy_only += 1
		elif series_filled_npy[i]==0 and series_filled_sps[i]!=0:
			count_sps_only += 1
		elif series_filled_npy[i]==0 and series_filled_sps[i]==0:
			count_bothmiss += 1
	print('number of both detection: %d'%count_both)
	print('number of sps only detection: %d'%count_sps_only)
	print('number of npy only detection: %d'%count_npy_only)
	print('number of all missed detection: %d'%count_bothmiss)


if __name__=='__main__':
	main()
