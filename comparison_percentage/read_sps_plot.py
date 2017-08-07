import sys
import numpy as np

def plot(structured_array, basenm, time_low, time_high):
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(13,3))
	for ele in structured_array:
		if ele['time']>=time_low and ele['time']<=time_high:
			plt.scatter(ele['time'], ele['dm'], s=ele['sigma']/3, color='black')
	axes = plt.gca()
	axes.set_xlim([time_low,time_high])
	axes.set_ylim([10,40])
	plt.xlabel('Arrival Time ($s$)')
	plt.ylabel('DM ($pc$ $cm^{-3}$)')
	plt.title('Pulses Found by PRESTO at DM %.1f $pc$ $cm^{-3}$'%round(structured_array['dm'][0],1))
	#print('1111')
	fig.savefig('%s_sps_DM%.1f_%d_%d.png'%(basenm, structured_array['dm'][0], time_low, time_high))
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
    plt.title('PRESTO Neigbhouring Detections Time Separation')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.xticks(np.arange(0,8,0.7))
    axes = plt.gca()
    axes.set_ylim([0,200])
    plt.show()


def main():
	infilenm = sys.argv[1]
	basenm = infilenm.split('.')[0]
	with open(infilenm) as f:
		lines = f.readlines()
	num_total = len(lines)
	print(lines[288])
	print(num_total)
	x = np.zeros(num_total-1, dtype=[('dm',np.float32),('sigma',np.float32),('time',np.float32),('sample',np.int16),('downfact',np.int16)])
	for i in range(num_total-1):
		curr_line = lines[i+1]
		entry = curr_line.split()
		dm, sigma, time, sample, downfact = round(float(entry[0].strip()),2), round(float(entry[1].strip()),2), round(float(entry[2].strip()),4), \
											entry[3].strip(), entry[4].strip()
		x[i] = (dm, sigma, time, sample, downfact)
	x = x[np.argsort(x['time'])]
	
	#plot(x, basenm, 0, 200)
	plot(x, basenm, 0, 410)
	#time_histo(x['time'])

if __name__ == "__main__":
    main()
