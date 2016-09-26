----------------------------------------------------------------------
print '==> defining test procedure'

minValid = 0.75
stability = 0
-- test function
function test()
   -- local vars
   local time = sys.clock()
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   -- set total error table
   totErr = torch.Tensor(1):fill(0)
   ftest = 0
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testInp[1]:size(1) do
      -- disp progress
      xlua.progress(t, testInp[1]:size(1))
      -- get new sample
      local input = {}
      for qw =1,table.getn(testInp) do
          table.insert(input, testInp[qw][{{t},{},{},{}}])
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
      local target = {}
      if opt.outType =='TSSseq' then
          target =  testOut[{{t},{},{},{}}]:squeeze():cuda()
      else
          target =  testOut[{{t},{}}]
      end
      -- test sample
      local pred = {}
      local tmpoutput = {}
      if iscomposite then
          if table.getn(input)==1 then
              tmpoutput = pretModel:forward(input[1])
          else
              tmpoutput = pretModel:forward(input)
          end
          pred = model:forward(tmpoutput):squeeze()
      else
          if table.getn(input)==1 then
              pred = model:forward(input[1]):squeeze()
          else
              pred = model:forward(input):squeeze()
          end
      end
      local err = 0
      if opt.type == 'double' then
          err = criterion:forward(pred, target:double())
          ftest = ftest + err
      elseif opt.type =='cuda' then
          err = criterion:forward(pred, target:cuda())
          ftest = ftest + err
      end
      mx,idx=pred:view(pred:nElement()):max(1)
      mx,jdx=target:view(target:nElement()):max(1)
      idx = idx:double()
      jdx = jdx:double()
      idx = math.ceil(idx[1]*10/width)
      jdx = math.ceil(jdx[1]*10/width)
      confusion:add(idx, jdx)
    end
    if ftest then
        totErr:add(ftest)
    end
      -- timing
    time = sys.clock() - time
    time = time / testInp[1]:size(1)
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    print(confusion)
    print("Mean test set error(100x):   " .. tostring(100*totErr[1])/testInp[1]:size(1))
  -- update log/plot
   testLogger:add{['% mean test set error'] = totErr[1]/testInp[1]:size(1)}
   testLogger2:add{['% mean test set accuracy'] = confusion.averageValid*100}


   if totErr[1] > opt.stopThresh * minValid then
     stability = stability + 1
   else
     stability = 0
   end
   if totErr[1]< minValid and stability > 5 then
     minValid =  totErr[1]
     local filename = paths.concat('../results/',opt.runName, 'modelMaxValid.net')
     os.execute('mkdir -p ' .. sys.dirname(filename))
     print('==> saving maxValid model to '..filename)
     torch.save(filename, model)
   end
   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   -- next iteration:
   confusion:zero()
end
