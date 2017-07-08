require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods


----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- In case of TSS represenatation, we divide the region into 10 classes
-- Just to get a feeling about the accuracy of the predictions. This is not a classification task,
-- therefore, these classes are not used for calculating the loss function. Loss function is calculated
-- by the relative entropy between the prediction and the target probability distributions.
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat('../results/',opt.runName, 'trainErr.log'))
testLogger = optim.Logger(paths.concat('../results/',opt.runName, 'testErr.log'))
trainLogger2 = optim.Logger(paths.concat('../results/',opt.runName, 'train.log'))
testLogger2 = optim.Logger(paths.concat('../results/',opt.runName, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'
maxValid = 0
stability = 0
function train()
   -- epoch tracker
   epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()
   if iscomposite then
      pretModel:evaluate()
    end
   -- set total error table
   totErr = torch.Tensor(1):fill(0)
   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainInp[1]:size(1),opt.batchSize do
        -- disp progress
        xlua.progress(t, trainInp[1]:size(1))
        -- create mini batch
        local inputs = {}
        local targets ={}
        for i = t,math.min(t+opt.batchSize-1,trainInp[1]:size(1)) do
             -- load new sample
            local input = {}
            for qw =1,table.getn(trainInp) do
                table.insert(input, trainInp[qw][{{shuffle[i]},{},{},{}}])
            end
            local target = {}
            if opt.outType == 'TSSseq' then
               target = trainOut[{{shuffle[i]},{},{},{}}]
            else
               target= trainOut[{{shuffle[i]},{}}]
            end
            if opt.type == 'double' then
               for qw =1,table.getn(input) do
                 input[qw] = input[qw]:double()
               end
            elseif opt.type == 'cuda' then
               for qw =1,table.getn(input) do
                 input[qw] = input[qw]:cuda()
               end
            end
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
             -- get new parameters
             if x ~= parameters then
                parameters:copy(x)
             end
             -- reset gradients
             gradParameters:zero()
             -- f is the average of all criterions
             local f = 0
             -- evaluate function for complete mini batch
             for i = 1,#inputs do
                 local output = {}
                 local tmpoutput ={}
                 if iscomposite then
                      if table.getn(inputs[i])==1 then
                        tmpoutput = pretModel:forward(inputs[i][1])
                      else
                        tmpoutput = pretModel:forward(inputs[i])
                      end
                        output = model:forward(tmpoutput)
                  else
                      if table.getn(inputs[i])==1 then
                        output = model:forward(inputs[i][1])
                      else
                        output = model:forward(inputs[i])
                      end
                  end
                    -- Detecting and removing NaNs
                    if output:ne(output):sum() > 0 then
                        print(sys.COLORS.red  .. ' weights has NaN/s')
                        output[output:ne(output)] = torch.log(1/width)
                    end
                    local err = {}
                    local df_do = {}
                    if opt.type == 'double' then
                        err = criterion:forward(output, targets[i]:double())
                        f = f + err
                        df_do = criterion:backward(output, targets[i]:double())
                    elseif opt.type =='cuda' then
                        err = criterion:forward(output:squeeze(), targets[i]:squeeze():cuda())
                        f = f + err
                        df_do = criterion:backward(output:squeeze(), targets[i]:squeeze():cuda())
                    end
                    if iscomposite then
                        if table.getn(inputs[i])==1 then
                            model:backward(tmpoutput, df_do)
                        else
                            model:backward(tmpoutput, df_do)
                        end
                    else
                        if table.getn(inputs[i])==1 then
                            model:backward(inputs[i][1], df_do)
                        else
                            model:backward(inputs[i], df_do)
                        end
                    end
                    mx,idx=output:view(output:nElement()):max(1)
                    mx,jdx=targets[i]:view(targets[i]:nElement()):max(1)
                    idx = idx:double()
                    jdx = jdx:double()
                    idx = math.ceil(idx[1]*10/width)
                    jdx = math.ceil(jdx[1]*10/width)
                    -- update confusion
                    confusion:add(idx, jdx)
                end
                     -- normalize gradients and f(X)
                    gradParameters:div(#inputs)
                    --  f = f/#inputs
                    totErr:add(f)
                     -- return f and df/dX
                    return f,gradParameters
          end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
  end
   -- time taken
   time = sys.clock() - time
   time = time / trainInp[1]:size(1)
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   -- print confusion matrix
   print(confusion)
   print('Mean train set error(100x):  ' .. tostring(100*totErr[1]/trainInp[1]:size(1)))

   -- update logger/plot
   trainLogger:add{['% mean train set error'] = totErr[1]/trainInp[1]:size(1)}
   trainLogger2:add{['% mean train set error'] = confusion.averageValid*100 }
   -- save/log current net
   local filename = paths.concat('../results/',opt.runName, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
