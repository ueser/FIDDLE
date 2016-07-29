require 'xlua'    -- xlua provides useful tools, like progress bars

----------------------------------------------------------------------
print '==> defining prediction procedure'

function predict()
   -- set the time
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- initialize the tensors for prediction recordings
   predictionInfo= torch.Tensor(predInp[1]:size(1),5):fill(0)
   predictionScores= torch.Tensor(predInp[1]:size(1),predInp[1]:size(4)):fill(0)

   for t = 1,predInp[1]:size(1) do
      -- disp progress
      xlua.progress(t, predInp[1]:size(1))
      -- get new sample
      local input = {}
      for qw =1,table.getn(predInp) do
          table.insert(input, predInp[qw][{{t},{},{},{}}])
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

      local pred = {}
      local tmpoutput = {}
      if iscomposite then
          if table.getn(input)==1 then
              tmpoutput = pretModel:forward(input[1])
          else
              tmpoutput = pretModel:forward(input)
          end
          pred = model:forward(tmpoutput):squeeze():double()
      else
          if table.getn(input)==1 then
              pred = model:forward(input[1]):squeeze():double()
          else
              pred = model:forward(input):squeeze():double()
          end
      end

      mx,idx=pred:view(pred:nElement()):max(1)
      idx = idx:double()

      predictionScores[t] = pred
      H = torch.cmul(pred,torch.exp(pred)):sum()
      -- 1. chromosome 2. strand 3.annotation index 4. tss position 5. entropy of prediction
      predictionInfo[t] = torch.Tensor({info[{{t},{1}}]:squeeze(),info[{{t},{2}}]:squeeze(),info[{{t},{3}}]:squeeze(),info[{{t},{4}}]:squeeze(),H})
   end

      -- timing
   time = sys.clock() - time
   time = time / predInp[1]:size(1)
   print("\n==> time to predict 1 sample = " .. (time*1000) .. 'ms')

   local f2 = hdf5.open('../results/' .. opt.runName .. '/prediction' .. opt.tag .. '.hdf5', 'w')
   f2:write('predictionScores', predictionScores)
   f2:write('predictionInfo', predictionInfo)
   f2:close()
end
