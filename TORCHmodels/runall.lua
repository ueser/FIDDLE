require 'torch';
require 'paths';

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('deepTSS main')
cmd:text()
cmd:text('Options:')
cmd:option('-runName', 'experiment', 'Each experiment (or run) shoud have a different name')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-source', '', 'Data source as an hdf5 file')
cmd:option('-size', 'small', 'how many samples do we load: small | full ')
cmd:option('-dataset', 'NS', 'Which datasets? : NS for NETseq | MS for MNaseSeq | DS for DNA | RS for RNAseq | TF for TFIIB-ChIPseq')
cmd:option('-outType', 'TSSseq', 'What type is the output? : TSSseq | annotation')
cmd:option('-unitNorm', 0, 'Should individual datapoints be normalized? : 1 | 0 ')
cmd:option('-globNorm', 0, 'Should dataset be globally normalized? : 1 | 0')
cmd:option('-smooth', 0, 'Should NETseq be smoothed? : 1 | 0')
-- model:
cmd:option('-rerun', 'none', 'Rerun from a trained state')
cmd:option('-predict', 'none', 'Prediction mode?')
cmd:option('-tag', '', 'tag output')
cmd:option('-numFilters', 40, 'Number of filters')
cmd:option('-filterWidth', 10, 'Width of the filters')
-- loss:
cmd:option('-loss', 'KLdist', 'type of loss function to minimize: KLdist | nll | mse | margin')
-- training:
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 0.1, 'learning rate at t=0')
cmd:option('-stopThresh', .9, 'Ratio of max validation as a stopping condition that is achieved stability')
cmd:option('-batchSize', 10, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- Make a separate folder for each experiment
if not paths.dirp('../results/' .. opt.runName) then
  paths.mkdir('../results/' .. opt.runName)
end

-- Save parameters as a text file for each experiment into its folder
paramFile = '../results/' .. opt.runName .. '/runParameters.txt'
local file = assert( io.open(paramFile,'w+') )

file:write('Options:\n\n')
for feat,val in pairs(opt) do
file:write(feat .. ': ' .. tostring(val) .. '\n')
end
file:close()

----------------------------------------------------------------------
print '==> executing all'


if opt.predict ~= 'none' then
    dofile 'getDataPred.lua'
    opt.rerun = true
else
    dofile 'getData.lua'
end

if opt.rerun ~='none' then
    iscomposite = true
    numModality=0
    if string.find(opt.dataset,'NS') then
        numModality = numModality +1
        opt.NSmodel = '../results/NSmodel/model.net'
    end
    if string.find(opt.dataset,'MS') then
        numModality = numModality +1
        opt.MSmodel = '../results/MSmodel/model.net'
    end
    if string.find(opt.dataset,'DS') then
        numModality = numModality+1
        opt.DSmodel = '../results/DSmodel/model.net'
    end
    if string.find(opt.dataset,'RS') then
        numModality = numModality +1
        opt.RSmodel = '../results/RSmodel/model.net'
    end
    if string.find(opt.dataset,'TF') then
        numModality = numModality +1
        opt.TFmodel = '../results/TFmodel/model.net'
    end
    if opt.predict ~= 'none' then
      dofile 'transferModules.lua'
      model = torch.load('../results/' .. opt.runName .. '/model.net')
    else
      dofile 'compositeModel.lua'
    end

else
    if opt.dataset == 'DS' then
        print '==> setting DS model'
        dofile 'models/DSmodel.lua'
    elseif opt.dataset == 'NS' then
        print '==> setting NS model'
        dofile 'models/NSmodel.lua'
    elseif opt.dataset == 'MS' then
        print '==> setting MS model'
        dofile 'models/MSmodel.lua'
    elseif opt.dataset == 'RS' then
        print '==> setting RS model'
        dofile 'models/RSmodel.lua'
    elseif opt.dataset == 'TF' then
        print '==> setting TF model'
        dofile 'models/TFmodel.lua'
    end
end

if opt.predict ~= 'none' then
    dofile 'predictModel.lua'
    predict()
else
    -- Set the loss function
    criterion = nn.DistKLDivCriterion()

    dofile 'trainModel.lua'
    dofile 'testModel.lua'

    torch.save('../results/' .. opt.runName .. '/options.t7',opt)
    ----------------------------------------------------------------------
    print '==> training!'

    writeCondition = false
    while true do
           train()
           test()
    end
end
