require 'nngraph'      -- provides all sorts of trainable modules/layers
----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> define parameters'
-----------------------------------------------------
-- Model Parameters
-----------------------------------------------------

dofile 'transferModules.lua'

nfeats = numModality
nstates = 20
filtSizeW = 10
filtSizeH=1
width=500
finalNout=500


prevOut = nn.Identity()()
--
mAll = nn.SpatialConvolutionMM(nfeats,nstates,filtSizeW,filtSizeH)(prevOut)
mAll = nn.ReLU()(mAll)
local Nshape = width-filtSizeW+1
mAll = nn.View(Nshape*nstates)(mAll)
mAll = nn.Linear(Nshape*nstates,1000)(mAll)
mAll = nn.Dropout(0.5)(mAll)
mAll = nn.Linear(1000,finalNout)(mAll)
mAll = nn.LogSoftMax()(mAll)

model = nn.gModule({prevOut},{mAll})
model:cuda()
