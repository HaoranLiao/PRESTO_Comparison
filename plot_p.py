import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import frb_L2_L3

def main():
	datafile = sys.argv[1]
	events = pickle.load(open('%s'%datafile))
	events = np.hstack([e.l1_headers for e in events])
	dm_min = events['dm']
	arrival_times = events['time'].astype(float)/1e6
	dm_max = events['dm'] + frb_L2_L3.L1_event.tree_ddm(events['itree'])
	fig = plt.figure()
	plt.errorbar(arrival_times, dm_min+(dm_max-dm_min)/2,\
	             yerr=(dm_max-dm_min)/2, marker='.', c='r', linestyle='None', markersize='3', elinewidth=1, capsize=0)
	plt.title('Pulses')
	plt.xlabel('Arrival Time ($s$)')
	plt.ylabel('DM ($pc$ $cm^{-3}$)')
	axes = plt.gca()
	axes.set_xlim([0,100])
	axes.set_ylim([0,50])
	fig.savefig('%s_single_pulses.png'%datafile)
	plt.show()

if __name__ == "__main__":
    main()

