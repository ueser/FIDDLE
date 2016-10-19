require 'hdf5';
require 'torch'
require 'nn'


-- Some functions

function smoothData(dat,nw)
  local nw = nw or 50
  sz = dat:size()
  newdat = dat:clone()
  for w = nw/2+1,sz[4]-nw/2 do
      newdat[{{},{},{},{w}}]=dat[{{},{},{},{w-nw/2,w+nw/2}}]:mean(4)
  end
  return newdat
end
function globalNormData(dat, mn, sd)
  dat[dat:ne(dat)] =0

  local mn = mn or dat:mean()
  dat:add(-mn)
  local sd = sd or dat:std()
  dat:div(sd)
  return dat, mn, sd
end
function propNomData(dat)
  dat:add(-dat:min())
  return dat:div(dat:sum())
end

-------------------------------------------------------------------
-- <====To be generalized ====> --
-- training/test size

if opt.size == 'extra' then
   print '==> using extra training data'
   trsize = 250000
   tesize = 10000
elseif opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 50000
   tesize = 5000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 1000
   tesize = 100
end
-- ====>To be generalized<==== --

----------------------------------------------------------------------
print '==> loading dataset'

sourcePath = '../data/hdf5datasets/' .. opt.source

local myFile = hdf5.open(sourcePath, 'r')
sz = myFile:read('NStrainInp'):dataspaceSize()
sz2 = myFile:read('NStestInp'):dataspaceSize()

-- Be careful here, if the hdf5 data contain 0s at the end, then it will be a problem for training. Make sure the end of the
-- data doesn't contain full zeros. Or change the randomization below...
J = math.random(1, sz[1]-trsize)
J2 =math.random(1, sz2[1]-tesize)
trainInp = {}
testInp = {}
if string.find(opt.dataset,'NS') then
    local NStrainInpTmp = myFile:read('NStrainInp'):partial({J,J+trsize-1},{1,sz[2]},{1,sz[3]},{1,sz[4]})
    local NStestInpTmp = myFile:read('NStestInp'):partial({J2,J2+tesize-1},{1,sz2[2]},{1,sz2[3]},{1,sz2[4]})
    if opt.smooth ==1 then
       NStrainInpTmp =smoothData(NStrainInpTmp)
       NStestInpTmp =smoothData(NStestInpTmp)
    end
    if opt.unitNorm == 1 then
       sums = torch.sum(NStrainInpTmp,4):squeeze():add(1)
       NStrainInpTmp[{{},{},{},{}}]:cdiv(sums:view(sums:size(1),2,1,1):expandAs(NStrainInpTmp))
       sums = torch.sum(NStestInpTmp,4):squeeze():add(1)
       NStestInpTmp[{{},{},{},{}}]:cdiv(sums:view(sums:size(1),2,1,1):expandAs(NStestInpTmp))
    end
    if opt.globNorm ==1 then
       NStrainInpTmp,mn,sd = globalNormData(NStrainInpTmp)
       NStestInpTmp,mn,sd = globalNormData(NStestInpTmp)
       opt.meanNS = mn
       opt.sdevNS = sd
    end
    table.insert(trainInp,NStrainInpTmp)
    table.insert(testInp,NStestInpTmp)
    print('NS is included')
end
if string.find(opt.dataset,'MS') then
    local MStrainInpTmp = myFile:read('MStrainInp'):partial({J,J+trsize-1},{1,sz[2]},{1,sz[3]},{1,sz[4]})
    local MStestInpTmp = myFile:read('MStestInp'):partial({J2,J2+tesize-1},{1,sz2[2]},{1,sz2[3]},{1,sz2[4]})
    MStrainInpTmp,mn,sd = globalNormData(MStrainInpTmp)
    MStestInpTmp = globalNormData(MStestInpTmp,mn,sd)
    table.insert(trainInp,MStrainInpTmp)
    table.insert(testInp,MStestInpTmp)
    opt.meanMS = mn
    opt.sdevMS = sd
    print('MS is included')
end
if string.find(opt.dataset,'DS') then
    sz3 = myFile:read('DStrainInp'):dataspaceSize()
    local DStrainInpTmp = myFile:read('DStrainInp'):partial({J,J+trsize-1},{1,sz3[2]},{1,sz3[3]},{1,sz3[4]})
    local DStestInpTmp = myFile:read('DStestInp'):partial({J2,J2+tesize-1},{1,sz3[2]},{1,sz3[3]},{1,sz3[4]})
    table.insert(trainInp,DStrainInpTmp)
    table.insert(testInp,DStestInpTmp)
    print('DS is included')
end
if string.find(opt.dataset,'RS') then
    sz3 = myFile:read('RStrainInp'):dataspaceSize()
    local RStrainInpTmp = myFile:read('RStrainInp'):partial({J,J+trsize-1},{1,sz3[2]},{1,sz3[3]},{1,sz3[4]})
    local RStestInpTmp = myFile:read('RStestInp'):partial({J2,J2+tesize-1},{1,sz3[2]},{1,sz3[3]},{1,sz3[4]})
    RStrainInpTmp,mn,sd = globalNormData(RStrainInpTmp)
    RStestInpTmp = globalNormData(RStestInpTmp,mn,sd)
    table.insert(trainInp,RStrainInpTmp)
    table.insert(testInp,RStestInpTmp)
    opt.meanRS = mn
    opt.sdevRS = sd
    print('RS is included')
end
if string.find(opt.dataset,'TF') then
    sz3 = myFile:read('TFtrainInp'):dataspaceSize()
    local TFtrainInpTmp = myFile:read('TFtrainInp'):partial({J,J+trsize-1},{1,sz3[2]},{1,sz3[3]},{1,sz3[4]})
    local TFtestInpTmp = myFile:read('TFtestInp'):partial({J2,J2+tesize-1},{1,sz3[2]},{1,sz3[3]},{1,sz3[4]})
    TFtrainInpTmp,mn,sd = globalNormData(TFtrainInpTmp)
    TFtestInpTmp = globalNormData(TFtestInpTmp,mn,sd)
    table.insert(trainInp,TFtrainInpTmp)
    table.insert(testInp,TFtestInpTmp)
    opt.meanTF = mn
    opt.sdevTF = sd
    print('TF is included')
end
sz = myFile:read('trainOut'):dataspaceSize()
if opt.outType == 'TSSseq' then
    trainOut = myFile:read('trainOut'):partial({J,J+trsize-1},{1,sz[2]},{1,sz[3]},{1,sz[4]})
    testOut = myFile:read('testOut'):partial({J2,J2+tesize-1},{1,sz[2]},{1,sz[3]},{1,sz[4]})
    -- print(testOut:size())
    print(trainOut)
else
    trainOut = myFile:read('trainOut'):partial({J,J+trsize-1},{1,sz[4]})
    testOut = myFile:read('testOut'):partial({J2,J2+tesize-1},{1,sz[4]})
end
    myFile:close()

collectgarbage()
