

module load dev/cuda/6.5.14
. ~/torch2/install/bin/torch-activate

# Global defaults
LR=1
MO=1e-2
BS=50


# Train individual models
: '
  for data in NS MS DS RS TF
  do
    toSubmit="th runall.lua -runName ${data}selected_LR${LR} -dataset $data -learningRate $LR -batchSize $BS -momentum $MO -source ScerForTraining.hdf5 -size full"
    echo $toSubmit
    bsub -R "select[ngpus]" -q short -J ${data} -W 11:59 -u umuteser@gmail.com -N ${toSubmit}
  done
'



# Train composite model

: '
for LR in 1 1e-1
do
    data=NSMSDSRSTF
    toSubmit="th runall.lua -runName LessParam${data}_LR${LR} -dataset $data -learningRate $LR -batchSize $BS -momentum $MO -source ScerForTraining.hdf5 -size full -rerun 1"
    echo $toSubmit
    bsub -R "select[ngpus]" -q short -J LessParam${data} -W 11:59 -u umuteser@gmail.com -N ${toSubmit}
  
done
'

# Predict a composite model


    data=NSMSDSRSTF
    toSubmit="th runall.lua -runName allModel -dataset $data -source DehaYACforPred.hdf5 -predict 1 -tag DhanYAC"
    bsub -R "select[ngpus]" -q short -J predict -W 11:59 -u umuteser@gmail.com -N ${toSubmit}

