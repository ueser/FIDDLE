LR=1e-1
MO=1e-3
NF=50
FS=5
#inputList=( 'NS' 'MS' 'DS' 'RS' 'CS' 'NS_MS_DS_RS_CS') 
inputList=( 'NS_MS_DS_RS_CS' )


module load dev/tensorflow/NoGPU
#module load dev/tensorflow/WithGPU

for input in "${inputList[@]}"
do

  toSubmit="python main.py --runName ${input}trained --testSize 1000 --inputs $input"
  echo $toSubmit

  bsub -R "rusage[mem=10000]" -q short -W 11:59 -J $input -u umuteser@gmail.com -N $toSubmit
done
#bsub -R "rusage[ngpus=1]" -q gpu -J "${runName}_gpu" -u umuteser@gmail.com -N $toSubmit

