module load dev/tensorflow/1.0-GPU

to_submit='python main.py --runName csnstsdsrs_ms_sng --batchSize 20 --learningRate 1e-4 --configuration config_ms.json'

bsub -q short -R "select[mem=15000]" -W 11:59 -J csnsdstsrs_ms_sng -u umuteser@gmail.com -N $to_submit

