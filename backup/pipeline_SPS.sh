#!/bin/bash

if [ "$1" == "" ]; then
  echo "$0: Please provide a directory name"
  exit 1
fi
if [ ! -d "$1" ]; then
  echo "$0: $1 is not a directory name"
  exit 1
fi

cd $1
echo $1
echo $2
single_pulse_search.py --fast *.dat > "sps_raw_result_$2.txt"
mv "sps_raw_result_$2.txt" ../
cd ..
exit 0


