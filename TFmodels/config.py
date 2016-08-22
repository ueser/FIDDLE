FLAGS = None

# list the inputs and the outputs name
# This should match with the hdf5 records.

def get_network_architecture(FLAGS):

    inputDict = {'NS':'NETseq',
                'MS':'MNaseseq',
                'RS':'RNAseq',
                'DS':'DNAseq',
                'CS':'ChIPseq',
                'TS':'TSSseq'}




    inputList=[]
    for kys in FLAGS.inputs.split('_'):
        inputList.append(inputDict[kys])

    outputList=[]
    for kys in FLAGS.outputs.split('_'):
        outputList.append(inputDict[kys])




    if FLAGS.restore:

        restore_dirs={inputDict[key]:'../results/'+key+'trained/' for key in inputList}
        for inputName in inputList:
            network_architecture[inputName] = pickle.load(open(restore_dirs[inputName]+'network_architecture.pkl','r'))
    else:
        inputHeights = {'DNAseq':4,'NETseq':2,'ChIPseq':2,'MNaseseq':2,'RNAseq':1}
        default_arch ={
        "inputShape": [4,500,1],
        "outputWidth": [500],
        "numberOfFilters":[80,80],
        "filterSize":[[2,5],[1,5]],
        "pool_size":[1,2],
        "pool_stride":[1,2],
        "FCwidth":1024,
        "dropout":0.5
        }
        for inputName in inputList:
            network_architecture[inputName] = default_arch.copy()
            network_architecture[inputName]['inputShape'][0]=inputHeights[inputName]
            network_architecture[inputName]['filterSize'][0][0]=inputHeights[inputName]

    return network_architecture, restore_dirs
