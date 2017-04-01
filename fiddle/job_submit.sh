module load dev/tensorflow/1.0-GPU

to_submit='python main.py --runName cnds_ts_singlestranded --learningRate 1e-4'

bsub -q short -R "rusage[ngpus=1]" -W 10:00 -J CNDS_TS_single -u umuteser@gmail.com -N $to_submit

