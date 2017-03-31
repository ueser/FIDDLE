module load dev/tensorflow/1.0-GPU

to_submit='python main.py --runName CN_DS2TS_lr1e-4 --learningRate 1e-4 --configuration CN2TS.config.json --dataDir ../data/hdf5datasets/CN2TS_WT_500bp_tss'

bsub -q short -R "rusage[ngpus=1]" -W 10:00 -J CN_DS2TS_gpu_1e4 -u umuteser@gmail.com -N $to_submit

