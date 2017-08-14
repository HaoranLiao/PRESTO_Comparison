import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.pylab
from scipy.optimize import curve_fit

def lin_func_no_intercept(x, k):
	return k*x

def readsps(infilenm, off):
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
		x[i] = (dm, sigma, time+off, sample, downfact)
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


def fill_t_series_sps(p, series, offset):
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
	series_filled.append(series[0]-offset)
	count = 1
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			num_p = int(round(diff/p))
			#print('num_p: %d'%num_p)
			series_filled.extend([0]*(num_p-1))
			series_filled.append(series[i]-offset)
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
            if array[i]-array[i-1]<0.5*0.714:
                dup_count += 1
    print(dup_count)
    n, bins, patches = plt.hist(diff, 25, range=[0,5],facecolor='black', align='mid')
    #plt.title('L1 Neigbhouring Detections Time Separation')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.xticks(np.arange(0,5,0.7))
    axes = plt.gca()
    axes.set_ylim([0,350])
    plt.savefig("spstimehisto.svg")
    plt.show()

def fill_full_series_npy(series_filled, series_full):
	#print(series_full['t'])
	x = np.zeros(len(series_filled), dtype=[('beam', np.int16), ('itree', np.int16), ('snr', np.float32), ('time', np.float32), ('dm_min', np.float32), ('dm_max', np.float32),\
								   ('dm_best_min', np.float32), ('dm_best_max', np.float32), ('dm_best', np.float32), ('grade', np.float32)])
	for i in range(len(series_filled)):
		if series_filled[i]!=0:
			#ind = series_full['t'].index(series_filled[i])
			#print('@@@@@@@@@@%f'%series_filled[i])
			ind = np.where((series_full['time']==series_filled[i]))
			ind = ind[0][0]
			#print(ind)
			x[i] = (series_full['beam'][ind], series_full['itree'][ind], series_full['snr'][ind], series_filled[i], series_full['dm_min'][ind], series_full['dm_max'][ind],\
				series_full['dm_best_min'][ind], series_full['dm_best_max'][ind], series_full['dm_best'][ind], series_full['grade'][ind],)
		else:
			x[i] = (0, 0, 0, series_filled[i], 0, 0, 0, 0, 0, 0)
	print('NPY full filled: \n\n\n')
	#print(x)
	series_filled_full = x
	return series_filled_full
	#t_series = np.arange(series[0])

def fill_full_series_sps(series_filled, series_full, offset):
	x = np.zeros(len(series_filled), dtype=[('dm',np.float32),('sigma',np.float32),('time',np.float32),('sample',np.int16),('downfact',np.int16)])
	for i in range(len(series_filled)):
		#ind = 0
		if series_filled[i]!=0:
			#ind = series_full['t'].index(series_filled[i])
			ind = np.where(((series_full['time']-offset)==series_filled[i]))
			ind = ind[0][0]
			#print(ind)
			#print(i)
			x[i] = (series_full['dm'][ind], series_full['sigma'][ind], series_filled[i], series_full['sample'][ind],\
					series_full['downfact'][ind])
			#ind += 1
		else:
			x[i] = (0, 0, series_filled[i], 0, 0)
	print('SPS full filled: \n\n\n')
	#print(x)
	series_filled_full = x
	return series_filled_full
	#t_series = np.arange(series[0])

def remove_npy_dup(data, p):
	data = data
	print(len(data['time']))
	ind = []
	for i in range(len(data)):
		if i>0:
			diff = data['time'][i]-data['time'][i-1]
			#print(diff)
			if diff<0.5*p:
				ind.append(i)
				#ind.append(i-1)
	data = np.delete(data, ind)
	print(len(data))
	return data

def plot_snr_compare(npy_fullresult, sps_fullresult):
	snr_npy_both = []
	snr_sps_both = []
	snr_npy_only = []
	snr_sps_only = []
	time_npy_both = []
	time_sps_both = []
	time_npy_only = []
	time_sps_only = []
	for i in range(len(npy_fullresult)):
		if npy_fullresult[i]['time']!=0 and sps_fullresult[i]['time']!=0:
			snr_npy_both.append(npy_fullresult[i]['snr'])
			snr_sps_both.append(sps_fullresult[i]['sigma'])
			time_npy_both.append(npy_fullresult[i]['time'])
			time_sps_both.append(sps_fullresult[i]['time'])
		elif npy_fullresult[i]['time']!=0 and sps_fullresult[i]['time']==0:
			snr_npy_only.append(npy_fullresult[i]['snr'])
			time_npy_only.append(npy_fullresult[i]['time'])
		elif npy_fullresult[i]['time']==0 and sps_fullresult[i]['time']!=0:
			snr_sps_only.append(sps_fullresult[i]['sigma'])
			time_sps_only.append(sps_fullresult[i]['time'])

	#print('The sps only: \n%s'%time_sps_only)

	diff_both = []
	for i in range(len(time_sps_both)):
		diff_both.append(snr_sps_both[i]-snr_npy_both[i])
	print(len(diff_both))
	print(len(time_sps_both))

	fig = plt.figure(figsize=(6,6))
	ax = plt.gca()
	#ax.set_ylim([0, 30])
	low_num = 1
	high_num = 20
	diff_ratio_both = []
	for i in range(len(time_sps_both)):
		diff_ratio_both.append(snr_sps_both[i]/25-snr_npy_both[i]/50)

	


	#plt.title('Single Pulse Search SNR at DM26.40 $pc cm^{-3}$ compared to L1 (Threshold 10)')
	
	#plt.title('L1 Detection SNR versus Single Pulse Search Detection SNR')
	
	#plt.scatter(time_sps_both[low_num:high_num], [x / 25 for x in snr_sps_both[low_num:high_num]], marker='x', c='black', linestyle='None', s=10, label='Single Pulse Search detections SNR')
	#plt.scatter(snr_sps_both[0:-2], snr_npy_both[1:-1], s=5)
	snr_both = plt.scatter(snr_sps_both, snr_npy_both, s=5, c='black')
	snr_sps = plt.scatter(snr_sps_only, [0]*len(snr_sps_only), s=120, c='blue', marker='|')
	snr_npy = plt.scatter([0]*len(snr_npy_only), snr_npy_only, s=120, c='green', marker='_')
	#plt.scatter(snr_npy_only)
	#plt.scatter(time_sps_only, snr_sps_only, marker='x', c='red', linestyle='None', s=10, label='Only detected by SPS')
	#plt.scatter(time_npy_both[low_num:high_num], [x / 50 for x in snr_npy_both[low_num:high_num]], marker='x', c='red', linestyle='None', s=10, label='L1 detection SNR')
	#plt.scatter(time_npy_both, diff_both, marker='x', c='red', linestyle='None', s=10, label='Only detected by SPS')

	plt.xlabel('Single Pulse Search Detection SNR')
	plt.ylabel('L1 Detection SNR')
	#print(np.corrcoef(snr_sps_both, snr_npy_both))[0, 1]
	plt.axvline(x=4, color='k', linestyle='--', lw=0.5)
	plt.axhline(y=8
		, color='k', linestyle='--', lw=0.5)
	ax.set_ylim([0,50])
	ax.set_xlim([0,50])
	print(time_sps_only)
	#for xc in time_npy_both[low_num:high_num]:
	#	plt.axvline(x=xc, color='k', linestyle='--', lw=0.5)

	# # calc the trendline
	# z = np.polyfit(snr_sps_both, snr_npy_both, 1)
	# p = np.poly1d(z)
	x = np.linspace(0, np.max(snr_sps_both))
	# plt.plot(x, p(x), "r--")
	# # the line equation:
	# print("y=%.6fx+(%.6f)"%(z[0],z[1]))

	popt, pcov = curve_fit(lin_func_no_intercept, snr_sps_both, snr_npy_both)
	perr = np.sqrt(np.diag(pcov))
	lin_fit = plt.plot(x, lin_func_no_intercept(x, *popt), 'r--', label='linear fit zero intercept')
	plt.legend(lin_fit, ['y=(%.2f$\pm$%.2f)$\cdot$x'%(popt[0], perr)], loc='upper right')

	plt.savefig("maskfrac.svg")

	plt.show()

	# fig = plt.figure()
	# plt.scatter(time_sps_both, diff_ratio_both, s=5)
	# plt.title('Difference between Single Pulse Search SNR Percentage and L1 SNR Percentage')
	# plt.axhline(y=-0.2, color='k', linestyle='--', lw=0.5)
	# plt.axhline(y=0.2, color='k', linestyle='--', lw=0.5)
	# #plt.show()
	# #for i in range(len(time_sps_both)):
		#print('%.1f    %.1f'%(time_sps_both[i],time_npy_both[i]))

	time_diff_sum = 0
	for i in range(len(time_sps_both)):
		time_diff_sum += (time_sps_both[i]-time_npy_both[i])**2

	return time_diff_sum

def remove_npy_otherdm(npy_result, lodm, hidm):
	#print(npy_result)
	ind = []
	for ele in npy_result:
		if ele['dm_best_min']>hidm or ele['dm_best_min']<lodm-2:
			for i in range(len(np.where(npy_result['time']==ele['time'])[0])):
				ind.append(np.where(npy_result['time']==ele['time'])[0][i])
	#print(ind)
	npy_result = np.delete(npy_result, ind)
	return npy_result


def main():
	from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	rc('font',**{'family':'serif','serif':['Times New Roman']})
	rc('text', usetex=True)


	time_diff_sum_array = []
	off = 0.475
	#for off in np.arange(0.0, 0.4, 0.1):

	sps_result = readsps(sys.argv[1], off)
	print('!!!!%f'%len(sps_result))
	npy_result = readnpy(sys.argv[2])
	#npy2_result = readnpy(sys.argv[3])

	#time_histo(sps_result['time'])
	npy_result = remove_npy_otherdm(npy_result, 26, 27)

	#time_histo(npy_result['time'])
	time_histo(sps_result['time'])

	offset = 0

	npy_result = remove_npy_dup(npy_result, 0.714)
	sps_result = remove_npy_dup(sps_result, 0.714)

	series_filled_sps, count_sps = fill_t_series_sps(0.71446, sps_result['time'], offset)
	series_filled_full_sps = fill_full_series_sps(series_filled_sps, sps_result, offset)
	series_filled_npy, count_npy = fill_t_series_npy(0.71446, npy_result['time'])
	series_filled_full_npy = fill_full_series_npy(series_filled_npy, npy_result)
	#series_filled_npy2, count_npy2 = fill_t_series_npy(0.71446, npy2_result['time'])

	print('length of the series_filled_sps: %d'%len(series_filled_sps))
	print('length of the full series_filled_sps: %d'%len(series_filled_full_sps))
	print('length of the series_filled_npy: %d'%len(series_filled_npy))
	print('length of the full series_filled_npy: %d'%len(series_filled_full_npy))
	#print('length of the series_filled_npy2: %d'%len(series_filled_npy2))

	count_both = 0
	count_bothmiss = 0
	count_npy_only = 0
	count_sps_only = 0
	for i in range(len(series_filled_sps)):
		#print('%.1f    %.1f'%(series_filled_sps[i], series_filled_npy[i]))
		print('%.1f    %.1f'%(series_filled_sps[i], series_filled_npy[i]))
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

	time_diff_sum = plot_snr_compare(series_filled_full_npy,series_filled_full_sps)

		
	# 	time_diff_sum_array.append('time offset %.3f diff sum %.3f'%(off, time_diff_sum))
	# 	#print(series_filled_full_npy['snr'])
	# print(time_diff_sum_array)

if __name__=='__main__':
	main()
