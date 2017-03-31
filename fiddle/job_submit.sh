module load dev/tensorflow/1.0-GPU

to_submit='python main.py --runName DNAseq_prediction_withAE --learningRate 1e-3'

bsub -q short -R "rusage[ngpus=1]" -W 10:00 -J DNAseq_predAE -u umuteser@gmail.com -N $to_submit

