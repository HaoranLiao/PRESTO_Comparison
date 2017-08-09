# PRESTO INSTALLATION GUIDE

## Environment
### Install Anaconda

```
wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
bash Anaconda2-4.3.0-Linux-x86_64.sh
source ~/.bashrc
```

## Libraries
### Install all libraries and software required except TEMPO

```
sudo apt-get install git libfftw3-bin libfftw3-dbg libfftw3-dev libfftw3-doc libfftw3-double3 libfftw3-long3 libfftw3-quad3 libfftw3-single3 pgplot5 csh automake gfortran libglib2.0-dev libccfits-dev libcfitsio3 libcfitsio3-dev libx11-dev libpng12-dev nvidia-cuda-dev libcuda1-331 -y
```
Note that it is important to have the dev versions of these packages.

### Intall TEMPO

```
git clone git://git.code.sf.net/p/tempo/tempo
cd tempo
```
```
./prepare
```
If this does not work, do
```
autoreconf –install
```
```
./configure
make
sudo make install
cp tempo.cfg src/
cp tempo.hlp src/
```

### Get PRESTO

```
cd ..
git clone git://github.com/scottransom/presto.git
cd presto
git pull
```

### Set environment variables

```
vi ~/.bashrc
```
Insert at the end of the file:
```
export PRESTO=YOUR_PATH_TO_PRESTO/presto
export PGPLOT=/usr/lib/pgplot5
export TEMPO=YOUR_PATH_TO_TEMPO/tempo/src
export FFTW_PATH=/usr
export PATH=$PATH:YOUR_PATH_TO_PRESTO/presto/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_PATH_TO_PRESTO /presto/lib
export PYTHONPATH=$PYTHONPATH:YOUR_PATH_TO_PRESTO/presto/python:YOUR_PATH_TO_PRESTO/presto/lib/python
export LD_LIBRARY_PATH=YOUR_PATH_TO_ANACONDA2/anaconda2/pkgs/libgcc-4.8.5-2/lib:YOUR_PATH_TO_ANACONDA2/anaconda2/lib:$LD_LIBRARY_PATH

```
Note, it is not PGPLOT=/usr/bin/pgplot5.
Exit the editor and activate the change,
```
source ~/.bashrc
```

### Install PRESTO

```
cd ../presto/src
``` 
or 
```
cd YOUR_PATH_TO_PRESTO/presto/src
```
```
make makewisdom (this takes some time)
make clean
make prep
make
```
If ever had this error:
```
~/anaconda2/bin/../lib/libgomp.so.1: version `GOMP_4.0' not found
```
Reason: Anaconda's gcc libs was compiled by gcc4.xx. by the system owned gcc version is gcc5.xx.
Do the following:
```
cp /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 YOUR_PATH_TO_ANACONDA2/anaconda2/pkgs/libgcc-4.8.5-2/lib
cp /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0 YOUR_PATH_TO_ANACONDA2/anaconda2/pkgs/libgcc-4.8.5-2/lib
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21 YOUR_PATH_TO_ANACONDA2/anaconda2/pkgs/libgcc-4.8.5-2/lib
cp /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 YOUR_PATH_TO_ANACONDA2/anaconda2/lib
cp /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0 YOUR_PATH_TO_ANACONDA2/anaconda2/lib
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21 $YOUR_PATH_TO_ANACONDA2/anaconda2/lib
```

## Test PRESTO
type in each of the following command to check:
```
rfifind
prepdata
prepsubband
DDplan.py
single_pulse_search.py
prepfold
accelsearch
```
