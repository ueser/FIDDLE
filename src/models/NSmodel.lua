--[[

NETseq  models

]]--

require 'torch'   -- torch
require 'nngraph'      -- provides all sorts of trainable modules/layers
----------------------------------------------------------------------

----------------------------------------------------------------------
print '==> define parameters'

-----------------------------------------------------
-- Model Parameters
-----------------------------------------------------


finalNout = 500
nfeats = 2 -- for sense and antisense for NSMS and should be 4 for DS
width = 500
height = 1
ninputs = nfeats*width*height
nstates = {40,80,500}
filtSizeW={10,10}
filtSizeH={1,1}
poolSize ={2,2}


-----------------------------------------------------
-- Model Architecture
-----------------------------------------------------


NSinp = nn.Identity()()

-- 1st Layer --
m11 = nn.SpatialConvolutionMM(nfeats,nstates[1],filtSizeW[1],filtSizeH[1])(NSinp):annotate{
   name = 'C1', description = 'Convolution',
   graphAttributes = {color = 'red'}
    }
m11 = nn.ReLU()(m11):annotate{
    name='NL1',description = 'ReLU'
}

m11 = nn.SpatialMaxPooling(poolSize[1],1,poolSize[1],1)(m11):annotate{
   name = 'P1', description = 'MaxPool'
    }

local Nshape = math.floor((width-filtSizeW[1]+1)/poolSize[1])

-- 2nd Layer --
m11 = nn.SpatialConvolutionMM(nstates[1],nstates[2],filtSizeW[2],filtSizeH[2])(m11):annotate{
   name = 'C2', description = 'Convolution',
   graphAttributes = {color = 'red'}
    }
m11 = nn.ReLU()(m11):annotate{
    name='NL2',description = 'ReLU'
}
m11 = nn.SpatialMaxPooling(poolSize[2],1,poolSize[2],1)(m11):annotate{
   name = 'P2', description = 'MaxPool'
    }

Nshape = math.floor((Nshape-filtSizeW[2]+1)/poolSize[2])
m11 = nn.View( nstates[2]*1*Nshape)(m11)
m11 = nn.Dropout(0.5)(m11)

-- 3rd Layer --
m11 = nn.Linear(nstates[2]*1*Nshape, finalNout)(m11):annotate{
    name='FC3',description = 'FullyConnected'
  }


-- LogSoftMax Layer --
m11 = nn.LogSoftMax()(m11)
model = nn.gModule({NSinp},{m11})
