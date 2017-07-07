import numpy as np
import sys

def readdat(infilenm):
	basenm = infilenm.split('.')[0]
	with open(infilenm) as f:
		lines = f.readlines()
	num_total = int(lines[-1].split()[0].strip())
	print(num_total)
	x = np.zeros(num_total, dtype=[('#',np.int32),('dm',np.float32),('snr',np.float32),('t',np.float32)])
	for i in range(num_total):
		curr_line = lines[i+1]
		entry = curr_line.split()
		index, dm, snr, t = int(entry[0].strip()), round(float(entry[1].strip()),2), round(float(entry[2].strip()),2), round(float(entry[3].strip()),4)
		x[i] = (index, dm, snr, t)
	#print(x['t'])
	x = x[np.argsort(x['t'])]
	#print(x)
	return x

def readrrattrap(infilenm, lodm, hidm):
	basenm = infilenm.split('.')[0]
	with open('rrattrap_testgroups.txt') as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	x = np.zeros(1, dtype=[('mindm',np.float32),('maxdm',np.float32),('ct',np.float32),('duration',np.float32),('maxsig',np.float32),('rank',np.float32)])
	count = 0
	for i in range(len(content)):
		if content[i].startswith('Group'):
			rank = round(float(content[i+6].split()[1].strip()),2)
			if rank==7.0 or rank ==6.0 or rank==5.0 or rank==4.0:
				mindm = round(float(content[i+1].split(':')[1].strip()),2)
				maxdm = round(float(content[i+2].split(':')[1].strip()),2)
				if mindm>=lodm and maxdm<=hidm:
					ct = round(float(content[i+3].split(':')[1].strip()),2)
					duration = round(float(content[i+4].split(':')[1].strip()),2)
					maxsig = round(float(content[i+5].split(':')[1].strip()),2)
					if count==0:
						x[count] = (mindm, maxdm, ct, duration, maxsig, rank)
					elif count>0:
						x = np.append(x, np.array([(mindm, maxdm, ct, duration, maxsig, rank)], dtype=x.dtype))
					count += 1
	#print(x['ct'])
	#print(count)
	x = x[np.argsort(x['ct'])]
	#print(x)
	return x

def compare(l1_res, rra_res):
	match = 0
	index_list = []
	for rra_pulse_t in rra_res['ct']:
		for i in range(len(l1_res['t'])):
			if rra_pulse_t>=l1_res['t'][i]-0.3 and rra_pulse_t<=l1_res['t'][i]+0.3:
				match += 1
				index_list.append(l1_res['#'][i])
				print('%.3f ----- %.3f'%(rra_pulse_t, l1_res['t'][i]))
				break
	seen = set()
	uniq = []
	for index in index_list:
		if index not in seen:
			uniq.append(index)
			seen.add(index)
		else:
			match -= 1
	print(match)
	
def compare2(l1_res, rra_res, err):
	match = 0
	dup = 0
	nomatch = 0
	ind_seen = set()
	rra_ct_match = []
	rra_ct_dup = []
	rra_ct_nomatch = []
	for rra_pulse_t in rra_res['ct']:
		for i in range(len(l1_res['t'])):
			if rra_pulse_t>=l1_res['t'][i]-err and rra_pulse_t<=l1_res['t'][i]+err:
				if l1_res['#'][i] not in ind_seen:
					match += 1
					ind_seen.add(l1_res['#'][i])
					rra_ct_match.append(round(rra_pulse_t,2))
					print('%.3f ----- %.3f'%(rra_pulse_t, l1_res['t'][i]))
					break
				else:
					dup += 1
					rra_ct_dup.append(round(rra_pulse_t,2))
					print('Duplicated: %.3f'%rra_pulse_t)
					break
			else:
				if i==len(l1_res['t'])-1:
					nomatch += 1
					rra_ct_nomatch.append(round(rra_pulse_t,2))
					print('nomatch: %.3f'%rra_pulse_t)
				else:
					continue
	print('Match: %d\nDup: %d\nNo match: %d'%(match,dup,nomatch))
	print(len(ind_seen))
	return {'rra_ct_match': rra_ct_match, 'rra_ct_dup': rra_ct_dup, 'rra_ct_nomatch': rra_ct_nomatch}

def plot(comp, l1, rra):
	import matplotlib.pyplot as plt
	import matplotlib.patches as mpatches
	fig = plt.figure()
	for rra_pulse in rra:
		cen_dm = rra_pulse['mindm']+(rra_pulse['maxdm']-rra_pulse['mindm'])/2
		if round(rra_pulse['ct'],2) in comp['rra_ct_match']:
			c = 'green'
		elif round(rra_pulse['ct'],2) in comp['rra_ct_dup']:
			c = 'blue'
		elif round(rra_pulse['ct'],2) in comp['rra_ct_nomatch']:
			c = 'red'
		#plt.errorbar(rra_pulse['ct'], cen_dm,\
		#		     xerr=rra_pulse['duration']/2, yerr=(rra_pulse['maxdm']-rra_pulse['mindm'])/2,\
		#			 marker='.', c=c, linestyle='None', markersize='3', elinewidth=1, capsize=0)
		plt.scatter(rra_pulse['ct'], cen_dm, marker='.', c=c, linestyle='None', s=3)
	plt.title('Rrattrap Grouped Pulses')
	plt.xlabel('Time (s)')
	plt.ylabel('DM ($pc$ $cm^{-3}$)')
	axes = plt.gca()
	axes.set_ylim([20.5,34.5])
	red_patch = mpatches.Patch(color='red', label='Not Matched')
	blue_patch = mpatches.Patch(color='blue', label='Duplicated Detection')
	green_patch = mpatches.Patch(color='green', label='Matched')
	plt.legend(handles=[red_patch, blue_patch, green_patch])
	#fig.savefig('example2data_rra_nobar.png')
	plt.show()

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

def time_histo_two(array1, array2):
	import matplotlib.pyplot as plt
	array1.sort()
	array2.sort()
	diff1 = []
	diff2 = []
	for i in range(len(array1)):
		if i>0:
			diff1.append(array1[i]-array1[i-1])
	n, bins, patches = plt.hist(diff1, 50, range=[0,10], alpha=0.6, align='mid', label='L1', facecolor='blue')
	for j in range(len(array2)):
		if j>0:
			diff2.append(array2[j]-array2[j-1])
	n, bins, patches = plt.hist(diff2, 50, range=[0,10], alpha=0.6, align='mid', label='PRESTO',facecolor='red')
	plt.title('Neighbouring Detections Time Separation')
	plt.xlabel('Time (s)')
	plt.ylabel('Count')
	plt.legend(loc='upper right')
	plt.xticks(np.arange(0,10,0.7))
	plt.show()


def fill_t_series_rra(p, series):
	#series = series_full['t']
	print(len(series))
	series_nodup = [series[0]]
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			if diff>p/2:
				series_nodup.append(series[i])
	series = series_nodup
	print(len(series))

	series_filled = []
	#series_std = []
	#series_std.append(series[0])
	series_filled.append(series[0]-0.2)
	count = 0
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
	print(count)
	series_std = np.arange(0,series[0]+(count)*p,p)
	num_zero_to_add_begin = int(round((series[0])/p))
	num_zero_to_add_end = 571-count-num_zero_to_add_begin-1
	#print(num_zero_to_add_end)
	series_filled = [0]*num_zero_to_add_begin+series_filled+[0]*num_zero_to_add_end
	#print(series_std)
	#print(series_filled)
	#print(len(series_filled))
	#print(len(series_std))
	return series_filled, count
	
def fill_t_series_l1(p, series):
	#series = series_full['t']
	print(len(series))
	series_nodup = [series[0]]
	for i in range(len(series)):
		if i>0:
			diff = series[i]-series[i-1]
			if diff>p/2:
				series_nodup.append(series[i])
	series = series_nodup
	print(len(series))

	series_filled = []
	#series_std = []
	#series_std.append(series[0])
	series_filled.append(series[0])
	count = 0
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
	series_std = np.arange(0,series[0]+(count+1)*p,p)
	num_zero_to_add = int(round((series[0])/p))
	series_filled = [0]*num_zero_to_add+series_filled
	#print(series_std)
	#print(series_filled)
	#print(len(series_filled))
	#print(len(series_std))
	return series_filled, count

def fill_full_series_l1(series_filled, series_full):
	#print(series_full['t'])
	x = np.zeros(len(series_filled), dtype=[('#',np.int32),('dm',np.float32),('snr',np.float32),('t',np.float32)])
	for i in range(len(series_filled)):
		if series_filled[i]!=0:
			#ind = series_full['t'].index(series_filled[i])
			#print('@@@@@@@@@@%f'%series_filled[i])
			ind = np.where((series_full['t']==series_filled[i]))
			ind = ind[0][0]
			#print(ind)
			x[i] = (i+1, series_full['dm'][ind], series_full['snr'][ind], series_filled[i])
		else:
			x[i] = (i+1, 0, 0, series_filled[i])
	print(x)
	series_filled_full = x
	return series_filled_full
	#t_series = np.arange(series[0])

def fill_full_series_rra(series_filled, series_full):
	x = np.zeros(len(series_filled), dtype=[('#',np.int32),('mindm',np.float32),('maxdm',np.float32),('ct',np.float32),('duration',np.float32),\
				('maxsig',np.float32),('rank',np.float32)])
	for i in range(len(series_filled)):
		#ind = 0
		if series_filled[i]!=0:
			#ind = series_full['t'].index(series_filled[i])
			ind = np.where(((series_full['ct']-0.2)==series_filled[i]))
			ind = ind[0][0]
			#print(ind)
			#print(i)
			x[i] = (i+1, series_full['mindm'][ind], series_full['maxdm'][ind], series_filled[i], series_full['duration'][ind],\
					series_full['maxsig'][ind], series_full['rank'][ind])
			#ind += 1
		else:
			x[i] = (i+1, 0, 0, series_filled[i], 0, 0, 0)
	print(x)
	series_filled_full = x
	return series_filled_full
	#t_series = np.arange(series[0])

def plot_snr_both(l1_ff, rra_ff, series_std):
	import matplotlib.pyplot as plt
	print(len(l1_ff))
	print(len(rra_ff))
	print(len(series_std))
	fig = plt.figure(figsize=(35, 5))
	ax1 = fig.add_subplot(111)
	both_no = []
	both_yes_t = []
	both_yes_snr = []
	l1_yes_t = []
	l1_yes_snr = []
	rra_yes_t = []
	rra_yes_snr =[]
	for i in range(len(l1_ff)):
		if l1_ff['t'][i]==0 and rra_ff['ct'][i]==0:
			both_no.append(series_std[i])
		elif l1_ff['t'][i]!=0 and rra_ff['ct'][i]==0:
			l1_yes_t.append(l1_ff['t'][i])
			l1_yes_snr.append(l1_ff['snr'][i])
		elif l1_ff['t'][i]==0 and rra_ff['ct'][i]!=0:
			rra_yes_t.append(rra_ff['ct'][i])
			rra_yes_snr.append(rra_ff['maxsig'][i])
		elif l1_ff['t'][i]!=0 and rra_ff['ct'][i]!=0:
			both_yes_t.append([l1_ff['t'][i],rra_ff['ct'][i]])
			both_yes_snr.append([l1_ff['snr'][i],rra_ff['maxsig'][i]])
		
	ax1.scatter(both_no,[1]*len(both_no),s=1, c='black',label='Both missed')
	ax1.scatter(l1_yes_t, l1_yes_snr, s=7, marker='o', c='blue',label='L1 grouped')
	ax1.scatter(rra_yes_t, rra_yes_snr,s=7, marker='x',c='green', label='rrattrap grouped')
	#ax1.scatter([row[0] for row in both_yes_t],[row[0] for row in both_yes_snr])
	#ax1.scatter([row[1] for row in both_yes_t],[row[1] for row in both_yes_snr])
	ax1.scatter([row[0] for row in both_yes_t],[row[0] for row in both_yes_snr],s=7,marker='o', c='red', label='Both detected - L1 grouped')
	ax1.scatter([row[1] for row in both_yes_t],[row[1] for row in both_yes_snr],s=7,marker='x', c='red', label='Both detected - rrattrap grouped')
	plt.legend(loc='upper right')
	plt.show()

			

def main():
	lodm, hidm = 0, 50
	l1_result = readdat(sys.argv[1])
	rrattrap_result = readrrattrap(sys.argv[2], lodm, hidm)
	#comp = compare2(l1_result, rrattrap_result, 0.2)
	#plot(comp, l1_result, rrattrap_result)
	#time_histo(rrattrap_result['ct'])
	#time_histo(l1_result['t'])
	time_histo_two(l1_result['t'], rrattrap_result['ct'])
	#print(l1_result['t'][0])
	#print(rrattrap['ct'][0])
	#series_filled_l1, count_l1 = fill_t_series_l1(0.71446, l1_result['t'])
	#series_filled_full_l1 = fill_full_series_l1(series_filled_l1, l1_result)
	#series_std = np.arange(0,len(series_filled_l1)*0.71446,0.71446)
	#series_filled_rra, count_rra = fill_t_series_rra(0.71446, rrattrap_result['ct'])
	#print(series_filled_rra)
	#print(rrattrap_result['ct'])
	#series_filled_full_rra = fill_full_series_rra(series_filled_rra, rrattrap_result)
	#plot_snr_both(series_filled_full_l1, series_filled_full_rra, series_std)

if __name__=='__main__':
	main()
