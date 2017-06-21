import os
import shutil
import subprocess
import numpy as np
import sys
import pickle
import traceback
import warnings
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEBUG = True

def dbgmsg(*args):
    if DEBUG:
        for arg in args:
            print("DEBUG:\t%s"%arg)

def is_number(s):
    try:
        float(s)
        return True

    except ValueError:
        return False

def run_readfile(filename):

    #Read data macros and store them in the header 
    
    header_raw = subprocess.check_output(["readfile", filename], shell=False)
    dbgmsg(header_raw)
    header = {}
    header_items = header_raw.split("\n")[4:37]
    for item in header_items:
        item_key = item.split("=")[0].strip()
        item_value = item.split("=")[1].strip()
        header[item_key] = item_value
    if DEBUG:
        for item in header:
            print(item+": "+header[item])
    
    return header

def run_rfifind(filename, rfifind_time_interval, *options):

    #Find RFI based on the rfifind_time_interval, and return the name of the mask file generated

    rfifind_output_name = filename.split('.')[0]
    option = ''
    for op in options:
	if is_number(op):
            option += str(op)+' '
	else:
	    option += op+' '        
    cmd = 'rfifind -time %.1f -o %s %s%s'\
            %(rfifind_time_interval, rfifind_output_name, option, filename)
    dbgmsg(cmd)
    subprocess.call(cmd, shell=True)
    maskname = filename.split(".")[0]+"_rfifind.mask"
    dbgmsg(maskname)
    
    return maskname

def run_DDplan(header, hidm, nsub, timeres):

    #Obtain the plan for de-dispersion and return it in the plan_header

    d = hidm                        			# highest DM to search
    n = header['Number of channels']
    b = header['Total Bandwidth (MHz)']
    t = str(float(header['Sample time (us)'])/(1e6))    # sample time in second
    f = header['Central freq (MHz)']
    s = nsub                        			# number of subbands
    r = timeres                     			# acceptable time resolution in ms
    dbgmsg('d=%s'%d, 'n=%s'%n, 'b=%s'%b,\
           't=%s'%t, 'f=%s'%f, 's=%s'%s, 'r=%s'%r)
    source_name = header['Source Name']
    current_path = os.getcwd()
    plan = subprocess.check_output(\
           ["DDplan.py",\
            '-d', d,\
            '-n', n,\
            '-b', b,\
            '-t', t,\
            '-f', f,\
            '-s', s,\
            '-r', r,\
            '-o', current_path+'/DDplan_plot_%s.ps'%source_name],\
            shell=False) 
    dbgmsg('DDplan: %s'%plan)

    plan_header = make_plan_header(plan)

    return plan_header
   
def make_plan_header(plan): 
    plan_header = {'Low DM': [],\
                   'High DM': [],\
                   'dDM': [],\
                   'DownSamp': [],\
                   'dSubDM': [],\
                   '#DMs': [],\
                   'DMs/call': [],\
                   'calls': [],\
                   'WorkFrac': []}

    #There may be multiple plans for corresponding DM intervals, read all of them separately
    #and store the ith plan in the ith entry of the header key's list
    for line in plan.split('\n'):
        if 'Low DM' in line:
	    firstline = plan.split('\n').index(line)+1
    plan_lines = plan.split('\n')[firstline:]
    plan_lines = filter(None, plan_lines)
    dbgmsg('plan_line: %s'%plan_lines)
    plan_value = []
    for line in plan_lines:
        plan_value.append(line.split())
    temp = []        
    for ele in plan_value:
        temp = temp+ele
    plan_value = temp
    dbgmsg('plan_value: %s'%plan_value)
    for i in range(len(plan_value)):
        if i%9==0:
            plan_header['Low DM'].append(plan_value[i])
        elif i%9==1:
            plan_header['High DM'].append(plan_value[i])
        elif i%9==2:
            plan_header['dDM'].append(plan_value[i])
        elif i%9==3:
            plan_header['DownSamp'].append(plan_value[i])
        elif i%9==4:
            plan_header['dSubDM'].append(plan_value[i])
        elif i%9==5:
            plan_header['#DMs'].append(plan_value[i])
        elif i%9==6:
            plan_header['DMs/call'].append(plan_value[i])
        elif i%9==7:
            plan_header['calls'].append(plan_value[i])
        else:
            plan_header['WorkFrac'].append(plan_value[i])

    return plan_header

def run_prepsubband(input_filename, plan_header, maskname, nsub):
    
    #Prepare subbands according to the plan_header
    #If the sub-plan has number of DMs small enough to handle, the code prepares that many subbands at once
    #If the sub-plan has number of DMs larger than 1000, prepare all subbands in chunk of 1000 (the system cannot handle more than that)
    #Move all .dat .inf with output_fileroot in the filename to the subfolder associated with the data

    output_fileroot = input_filename.split('.')[0]
    num_plan = int(len(plan_header['High DM']))
    dbgmsg('Number of DDplan: %s'%num_plan) 

    try:
        for i in range(num_plan):
            lodm = plan_header['Low DM'][i]
            dmstep = plan_header['dDM'][i]
            numdms = plan_header['#DMs'][i]
            downsamp = plan_header['DownSamp'][i]
            dbgmsg('lodm=%s'%lodm, 'dmstep=%s'%dmstep,\
               'numdms=%s'%numdms, 'downsamp=%s'%downsamp, 'nsub=%s'%nsub)

            #Prepare subbands in chunks of 1000 for this sub-plan
    	    if float(numdms)>1000:
                integer = int(numdms)//1000
                remainder = int(numdms)%1000
                for j in range(integer+1):
                    if j==integer: 
                        numdms = remainder+1
                    else:
                        numdms = 1000
                    subprocess.call(['prepsubband',\
                             	     '-lodm', str(float(lodm)+1000*j*float(dmstep)),\
                             	     '-dmstep', dmstep,\
                             	     '-numdms', str(numdms),\
                             	     '-downsamp', downsamp,\
                             	     '-nobary',\
                             	     '-nsub', nsub,\
                             	     '-o', output_fileroot, input_filename],\
                             	     #'-mask', maskname],\
                             	     shell=False)
                    dir = move_subbands(output_fileroot)
                 
            #If the number of DMs is smaller than 1000 in this subplan, prepare all subbands at once    
            else:
                subprocess.call(['prepsubband',\
                                 '-lodm', lodm,\
                                 '-dmstep', dmstep,\
                                 '-numdms', str(int(numdms)+1),\
                                 '-downsamp', downsamp,\
                                 '-nobary',\
                        	 '-nsub', nsub,\
                        	 '-o', output_fileroot, input_filename],\
                        	 #'-mask', maskname],\
                        	 shell=False)
		dir = move_subbands(output_fileroot)	    	
		
    except Exception:
        dir = move_subbands(output_fileroot)
	traceback.print_exc(file=sys.stdout)
	print('Subbands moved\nSearch interupted\nExit')
        sys.exit()
    except (KeyboardInterrupt, SystemExit):
        dir = move_subbands(output_fileroot)
        print('\nSubbands moved\nSearch inerupted\nExit')
	sys.exit()
    finally:
        return dir

def group_single_pulse_search_plot(input_filename, plan_header, dir, group):

    #Group is for single_pulse_search where the code generates folded plot of a group of DMs
    #If no group is given, do single pulse search on all .dat file in the working directory 
    #  and fold them all in the generated plot
    #The grouping can go across DMs with different DM steps in between
    #Rename the generate dgrouped plot according to the grouped low DM and high DM

    try:
        if group is None:
            run_single_pulse_search(dir)
	else:
	    os.chdir(dir)
            group =  round(float(group), 2)
	    output_fileroot = input_filename.split('.')[0]
	    num_dDM = len(plan_header['dDM'])
	    dm_range = round(float(plan_header['High DM'][-1]), 2)
            num_group = int(float(dm_range)//group)+1 
	    dDM = float(plan_header['dDM'][0])
	    dbgmsg('Number of subgroup: %s'%num_group, 'DM range: %s'%dm_range,\
               	   'Number of DM step: %s'%num_dDM)
	    temp = 0.00
	    for i in range(num_group):
		filenames = ''
		current_dm = temp
		current_lodm = current_dm
	        while (round(current_dm, 2)<=(i+1)*group and round(current_dm, 2)<=dm_range):
		    filenames = filenames+'*DM%.2f*.dat '%(current_dm)
		    for j in range(num_dDM-1):
			if round(current_dm, 2)==float(plan_header['High DM'][j]):
			    dDM = float(plan_header['dDM'][j+1])
			    print(round(current_dm, 2), dDM)
			    dbgmsg('DM step has changed')
		    dbgmsg(current_dm, dDM)
		    current_dm += dDM
	        temp = round(current_dm-dDM, 2)
		current_hidm = temp
	        run_single_pulse_search(dir, filenames)
	        os.rename(output_fileroot+'_singlepulse.ps',\
			  '%s_%.1f_%.1f_singlepulse.ps'%(output_fileroot, current_lodm, current_hidm))

    except ValueError:
        traceback.print_exc(file=sys.stdout)
        print('No data point written anymore after%s\nSearch finished'%current_hidm)
        sys.exit()

def run_single_pulse_search(dir, *args):

    #Do single pulse searching on all DM .dat file in the dir if no other input argument
    #Do single pulse searching on all DM .dat file specified in the args (which should be a string)
    #each single pulse seaching will generate a plot which groups all the input DM

    if len(args)==0:
        subprocess.call('single_pulse_search.py --fast *.dat', cwd=dir+'/', shell=True)
    else:
        for arg in args:
            subprocess.call('single_pulse_search.py --fast %s'%arg, cwd=dir+'/', shell=True)   

def move_subbands(fileroot):

    #Move all subbands to the subfloder with the same data name

    current_path = os.getcwd()
    dir = current_path+"/"+fileroot+"_subbands_nomask"
    if not os.path.exists(dir):
        os.makedirs(dir)
    files = os.listdir(current_path)
    for f in files:
        if f.startswith(fileroot+"_DM"):
            shutil.move(current_path+"/"+f, dir+"/"+f)
        elif f.endswith('singlepulse.ps'):
            shutil.move(current_path+"/"+f, dir+"/"+f)
    dbgmsg('subband_dir: %s'%dir)
 
    return dir

def get_single_pulse_search_macro(dir, fileroot):
    sps_result = {'File name': [],\
                  'DM': [],\
                  'Pseudo-median block std': [],\
                  'Percentage of bad blocks': [],\
                  'Number of pulse candidates': []}                 
    subprocess.check('single_pulse_search.py -f *.dat > sps_raw_result_%s.txt'%fileroot,\
                     cwd=dir+'/', shell=True)
    with open("sps_raw_result_%s.txt"%fileroot) as t:
       content = t.readlines()
    t.close()
    content = [x.strip() for x in content]
    i = 0
    for j in range(len(content)):
        if content[j].startswith('Reading'):
            sps_result['File name'].append(content[j].split('"')[1])
            sps_result['DM'].\
              append(float(sps_result['File name'][i].split('DM')[1].split('.dat')[0]))
            sps_result['Pseudo-median block std'].\
              append(float(content[j+2].split('=')[1].strip()))
            sps_result['Percentage of bad blocks'].\
              append(float(content[j+3].split('i.e.')[1].split('%')[0].strip()))
            sps_result['Number of pulse candidates'].\
              append(int(content[j+5].split('Found')[1].split('pulse')[0].strip()))
            i += 1    
    dbgmsg(sps_result)

    return sps_result   

def read_pulses(dir):
    os.chdir(dir)
    with open('rrattrap_testgroups.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    pulse = []
    count = 0
    for i in range(len(content)):
		info = {}
		if content[i].startswith('Group'):
			rank = float(content[i+6].split()[1].strip())
			if rank==7.0 or rank==6.0 or rank==5.0 or rank==4.0:
				info['Min DM'] = float(content[i+1].split(':')[1].strip())
				info['Max DM'] = float(content[i+2].split(':')[1].strip())
				info['Center time'] =  float(content[i+3].split(':')[1].strip())
				info['Duration'] = float(content[i+4].split(':')[1].strip())
				info['Max Sigma'] = float(content[i+5].split(':')[1].strip())
				info['Rank'] = rank
				pulse.append(info)
				count += 1
    dbgmsg(pulse, count)
   
    return pulse

def read_pulses_specific(dir, lodm, hidm, lotime, hitime):
	os.chdir(dir)
	with open('rrattrap_testgroups.txt') as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	pulse = []
	count = 0
	for i in range(len(content)):
		info = {}
		if content[i].startswith('Group'):
			rank = float(content[i+6].split()[1].strip())
			if rank==7.0 or rank ==6.0 or rank==5.0 or rank==4.0:
				info['Min DM'] = float(content[i+1].split(':')[1].strip())
				info['Max DM'] = float(content[i+2].split(':')[1].strip())
				info['Center time'] =  float(content[i+3].split(':')[1].strip())
				info['Duration'] = float(content[i+4].split(':')[1].strip())
				info['Max Sigma'] = float(content[i+5].split(':')[1].strip())
				info['Rank'] = rank
				if info['Center time']>=lotime and info['Center time']<=hitime\
				  and info['Min DM']>=lodm and info['Max DM']<=hidm:
					pulse.append(info)
					count +=1
	dbgmsg(pulse, count)

	return pulse

def plot_pulses(dir, pulses, fileroot):
	os.chdir(dir)
	fig = plt.figure()
	for pul in pulses:
		c = pul['Rank']
		#c = []
		#c.append(pul['Rank'])
		cen_dm = pul['Min DM']+(pul['Max DM']-pul['Min DM'])/2
		#c = ['red' if x==7.0 else 'blue' if x==4.0 else 'green' if x==5.0 else x for x in c]
		#c = ['yellow' if x==4.0 else x for x in c]
		if c==7.0:
			c = 'red'
		elif c==6.0:
			c = 'blue'
		elif c==5.0:
			c = 'green'
		elif c==4.0:
			c = 'yellow'
		else:
			c = 'white'
		plt.errorbar(pul['Center time'], cen_dm,\
					 xerr=pul['Duration']/2, yerr=(pul['Max DM']-pul['Min DM'])/2,\
					 marker='.', c=c, linestyle='None', markersize='3', elinewidth=1, capsize=0)
	plt.title('Single Pulses')
	plt.xlabel('Time (s)')
	plt.ylabel('DM Range')
	red_patch = mpatches.Patch(color='red', label='Rank 7')
	blue_patch = mpatches.Patch(color='blue', label='Rank 6')
	green_patch = mpatches.Patch(color='green', label='Rank 5')
	yellow_patch = mpatches.Patch(color='yellow', label='Rank 4')
	plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
	#fig.savefig('%s_single_pulses.png'%fileroot)
	plt.show()

def run_rrattrap(dir):
	subprocess.call('python rrattrap.py --use-configfile --use-DMplan --vary-group-size --inffile \
					../*rfifind.inf -o rrattrap_test *pulse', cwd=dir+'/', shell=True)

def save_obj(obj, name):
    with open(os.getcwd()+'/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load_obj(name):
    with open(os.getcwd()+'/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)      

def main():
	input_filename = sys.argv[1]
	fileroot = input_filename.split('.')[0]
	dash = '---------------'
    
	print("%sSTART READING DATA HEADER%s"%(dash, dash))
    #If there exists a data header file associated, read it; otherwise, generate header file
	if not os.path.isfile(fileroot+'_DataHeader.pkl'):
		header = run_readfile(input_filename)
		save_obj(header, fileroot+'_DataHeader')
	else:
		header = load_obj(fileroot+'_DataHeader')
	dbgmsg('Header: %s'%header)
	print("%sDONE READING%s%s\n"%(dash, dash, dash))

    #Source name is the telescope name
	source_name = header['Source Name']
	print('SOURCE: %s\n'%source_name)
 
    #If there exists a DDplan file associated, read it; otherwise, generate DDplan header file
    #hidm: DM upper limit(lower limit 0.00), nsub: number of subbands, timeres: acceptable time resolution in ms
	print("%sSTART DDPLAN%s"%(dash, dash))
	hidm = '1000'
	nsub = '128'
	timeres = '1'               
    #if not os.path.isfile(source_name+'_DDplanHeader.pkl'):
	plan_header = run_DDplan(header, hidm, nsub, timeres)
    #    save_obj(plan_header, source_name+'_DDplanHeader')
    #else:
    #    plan_header = load_obj(source_name+'_DDplanHeader')
	dbgmsg('Plan Header: %s'%plan_header, 'nsub=%s'%nsub)
	print("%sDONE DDPLAN%s\n"%(dash, dash))

    #If there exists the RFI files, especially the mask, read them; otherwise, generate the RFI files
    #rfifind_time_interval: time interval for maskng in seconds
	#clip: remove the time interval where the SNR is above clip at zero DM; that signal is considered RFI (default 6.0)
	#specify other options to run rfifind in the last input argument
	print("%sSTART FINDING RFI%s"%(dash, dash))
	rfifind_time_interval = 1.0
	clip = 6.0						
	#if not os.path.isfile(fileroot+'_rfifind.mask'):
	#maskname = run_rfifind(input_filename, rfifind_time_interval, '-clip %.1f'%clip)
	#else:
	maskname = fileroot+'_rfifind.mask'
	#dbgmsg('Mask Name: %s'%maskname)
	print("%sDONE FINDING RFI%s\n"%(dash, dash))
	
	#Prepare subbands
	print("%sSTART PREPARING SUBBANDS & SINGLE PULSE SEARCH%s"%(dash, dash))
	output_dir = run_prepsubband(input_filename, plan_header, maskname, nsub)
	#output_dir = '/home/presto/workspace/17-02-08-incoherent/frb_search_1/composition_p1_subbands'
	print("%sDONE SINGLE PULSE SEARCH & FILES MOVED%s\n"%(dash, dash))

	#Do single pulse search based on the gourp specified
	#group: the number fo DMs grouped together to generate the folded plot
	print("%sSTART PREPARING SUBBANDS & SINGLE PULSE SEARCH%s"%(dash, dash))
	group = 100
	group_single_pulse_search_plot(input_filename, plan_header, output_dir, group)
	print("%sDONE SINGLE PULSE SEARCH & FILES MOVED%s\n"%(dash, dash))

	print("%sSTART RRATTRAPS%s"%(dash, dash))
	run_rrattrap(output_dir)
	print("%sDONE RRATTRAPS%s\n"%(dash, dash))
	
	print("%sSTART GROUPING PULSES%s"%(dash, dash))
	pulses = read_pulses_specific(output_dir, 0.0, 50.0, 0.0, 500.0)
	print("%sDONE GROUPING PULSES%s\n"%(dash, dash))
	
	print("%sSTART %s"%(dash, dash))
	#plot_pulses(output_dir, pulses, fileroot)
	print("%sDONE GROUPING PULSES%s\n"%(dash, dash))


    
if __name__ == "__main__":
    main()
