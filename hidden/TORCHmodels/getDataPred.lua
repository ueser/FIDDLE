require 'hdf5';
require 'torch'
require 'nn'

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


----------------------------------------------------------------------
print '==> loading dataset'

if paths.filep('../data/hdf5datasets/' .. opt.source) then
  sourcePath = '../data/hdf5datasets/' .. opt.source
else
  sourcePath = opt.source
end


local myFile = hdf5.open(sourcePath, 'r')
      sz = myFile:read('NStrainInp'):dataspaceSize()
      predInp = {}

if string.find(opt.dataset,'NS') then
    local NStrainInpTmp = myFile:read('NStrainInp'):all()
    if opt.smooth ==1 then
      NStrainInpTmp =smoothData(NStrainInpTmp)
    end
    if opt.unitNorm == 1 then
      sums = torch.sum(NStrainInpTmp,4):squeeze():add(1)
      NStrainInpTmp[{{},{},{},{}}]:cdiv(sums:view(sums:size(1),2,1,1):expandAs(NStrainInpTmp))
    end
    if opt.globNorm ==1 then
      NStrainInpTmp,mn,sd = globalNormData(NStrainInpTmp)
      opt.meanNS = mn
      opt.sdevNS = sd
    end
    table.insert(predInp, NStrainInpTmp)
    print('NS is included')
end

if string.find(opt.dataset,'MS') then
    local MStrainInpTmp = myFile:read('MStrainInp'):all()
     MStrainInpTmp,mn,sd = globalNormData(MStrainInpTmp)
     table.insert(predInp, MStrainInpTmp)
     opt.meanMS = mn
     opt.sdevMS = sd
     print('MS is included')
end

if string.find(opt.dataset,'DS') then
    sz3 = myFile:read('DStrainInp'):dataspaceSize()
    local DStrainInpTmp = myFile:read('DStrainInp'):all()
    table.insert(predInp, DStrainInpTmp)
    print('DS is included')
end
if string.find(opt.dataset,'RS') then
    sz3 = myFile:read('RStrainInp'):dataspaceSize()
    local RStrainInpTmp = myFile:read('RStrainInp'):all()
    RStrainInpTmp,mn,sd = globalNormData(RStrainInpTmp)
    table.insert(predInp, RStrainInpTmp)
    opt.meanRS = mn
    opt.sdevRS = sd
    print('RS is included')
end
if string.find(opt.dataset,'TF') then
    sz3 = myFile:read('TFtrainInp'):dataspaceSize()
    local TFtrainInpTmp = myFile:read('TFtrainInp'):all()
    TFtrainInpTmp,mn,sd = globalNormData(TFtrainInpTmp)
    table.insert(predInp, TFtrainInpTmp)
    opt.meanTF = mn
    opt.sdevTF = sd
    print('TF is included')
end

info = myFile:read('info'):all():double()
       myFile:close()
collectgarbage()
