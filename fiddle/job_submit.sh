module load dev/tensorflow/1.0-GPU

to_submit='python main.py --runName CN_DS2TS --configuration CN2TS.config.json --dataDir ../data/hdf5datasets/CN2TS_WT_500bp_tss'

bsub -q short -W 10:00 -J CN_DS2TS_nogpu -u umuteser@gmail.com -N $to_submit

