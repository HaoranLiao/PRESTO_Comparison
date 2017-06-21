import sys
import numpy as np

def plot(t, dm, basenm):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.scatter(t, dm, s=2)
	axes = plt.gca()
	#axes.set_xlim([xmin,xmax])
	axes.set_ylim([0,50])
	plt.xlabel('Arrival Time ($s$)')
	plt.ylabel('DM ($pc$ $cm^{-3}$)')
	plt.title('Pulses Found by L1')
	print('1111')
	fig.savefig('%s_pulses.png'%basenm)
	plt.show()

def main():
	infilenm = sys.argv[1]
	basenm = infilenm.split('.')[0]
	with open(infilenm) as f:
		lines = f.readlines()
	num_total = int(lines[-1].split()[0].strip())
	print num_total
	x = np.zeros(num_total, dtype=[('#',np.int32),('dm',np.float32),('snr',np.float32),('t',np.float32)])
	for i in range(num_total):
		curr_line = lines[i+1]
		entry = curr_line.split()
		index, dm, snr, t = int(entry[0].strip()), round(float(entry[1].strip()),2), round(float(entry[2].strip()),2), round(float(entry[3].strip()),4)
		x[i] = (index, dm, snr, t)
	
	plot(x['t'], x['dm'], basenm)

if __name__ == "__main__":
    main()
