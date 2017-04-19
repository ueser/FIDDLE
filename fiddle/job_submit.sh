module load dev/tensorflow/1.0-GPU

to_submit='python main.py --runName csnsmsdsrs_ts_sng --batchSize 50 --learningRate 1e-3'

bsub -q short -R "rusage[ngpus=1]" -R "select[mem=20000]" -W 11:59 -J csnsnsmsrs_ts_sng -u umuteser@gmail.com -N $to_submit

