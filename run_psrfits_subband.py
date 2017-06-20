import rfifind
import numpy as np
import subprocess
import sys

def main():
	fitsfilenm = sys.argv[1]
	basenm = fitsfilenm.split('.')[0]
	dm = 0
	mask_subband = rfifind.rfifind("%s_rfifind.mask"%(basenm))  
	mask_subband.set_zap_chans(power=1000,plot=False)  
	mask_subband.set_weights_and_offsets()
	mask_subband.write_weights(filename="%s_weights.txt"%(basenm))
	chan, weights = np.loadtxt("%s_weights.txt"%(basenm), unpack = True, skiprows=1, dtype=int)
	rev_weights = weights[::-1]
	with open('%s_weights.txt'%(basenm), 'r') as f:
		header = f.readline()
	data = np.column_stack([chan,rev_weights])
	with open('%s_rev_weights.txt'%(basenm), 'w') as f:
		f.write(header)
		np.savetxt(f, data, fmt="%d", delimiter="\t")
	cmd = "psrfits_subband -dm %.1f -nsub 128 -o %s_subband_%.1f -weights %s_rev_weights.txt %s"%(dm,basenm,dm,basenm,fitsfilenm)
	subprocess.call(cmd, shell=True)
	print('end')
