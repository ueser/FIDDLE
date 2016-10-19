require 'nngraph'      -- provides all sorts of trainable modules/layers

-----------------------------------------------------
-- Model Architecture
-----------------------------------------------------
allInp = nn.Identity()()
numModality = 0
if string.find(opt.dataset,'NS') then
    NSmodel = torch.load(opt.NSmodel)
    numModality = numModality +1
    NSinp = nn.SelectTable(numModality)(allInp)
    for indexNode, node in ipairs(NSmodel.forwardnodes) do
        if node.data.module and indexNode<=NSmodel:size() then
          --to make sure, print the modules that are transferred
            print('transferred modules: ')
            print(node.data.module)
            if indexNode==2 then
                node.data.module:evaluate()
                mNS = node.data.module(NSinp)
            else
                node.data.module:evaluate()
                mNS = node.data.module(mNS)
            end
        end
    end
    print 'NS model copied'
end

if string.find(opt.dataset,'MS') then
    MSmodel = torch.load(opt.MSmodel)
    numModality = numModality +1
    MSinp = nn.SelectTable(numModality)(allInp)
    for indexNode, node in ipairs(MSmodel.forwardnodes) do
        if node.data.module and indexNode<=MSmodel:size() then
            if indexNode==2 then
                node.data.module:evaluate()
                mMS = node.data.module(MSinp)
            else
                node.data.module:evaluate()
                mMS = node.data.module(mMS)
            end
        end
    end
    print 'MS model copied'
end

if string.find(opt.dataset,'DS') then
    DSmodel = torch.load(opt.DSmodel)
    numModality = numModality +1
    DSinp = nn.SelectTable(numModality)(allInp)
    for indexNode, node in ipairs(DSmodel.forwardnodes) do
        if node.data.module and indexNode<=DSmodel:size() then
            if indexNode==2 then
                node.data.module:evaluate()
                mDS = node.data.module(DSinp)
            else
                node.data.module:evaluate()
                mDS = node.data.module(mDS)
            end
        end
    end
    print 'DS model copied'
end

if string.find(opt.dataset,'RS') then
    RSmodel = torch.load(opt.RSmodel)
    numModality = numModality +1
    RSinp = nn.SelectTable(numModality)(allInp)
    for indexNode, node in ipairs(RSmodel.forwardnodes) do
        if node.data.module and indexNode<=RSmodel:size() then
            if indexNode==2 then
                node.data.module:evaluate()
                mRS = node.data.module(RSinp)
            else
                node.data.module:evaluate()
                mRS = node.data.module(mRS)
            end
        end
    end
    print 'RS model copied'
end

if string.find(opt.dataset,'TF') then
    TFmodel = torch.load(opt.TFmodel)
    numModality = numModality +1
    TFinp = nn.SelectTable(numModality)(allInp)
    for indexNode, node in ipairs(TFmodel.forwardnodes) do
        if node.data.module and indexNode<=TFmodel:size() then
            if indexNode==2 then
                node.data.module:evaluate()
                mTF = node.data.module(TFinp)
            else
                node.data.module:evaluate()
                mTF = node.data.module(mTF)
            end
        end
    end
    print 'TF model copied'
end

print '==> models are loaded'

if string.find(opt.dataset,'NS') then
    mAll = mNS
end
if string.find(opt.dataset,'MS') then
    mAll = nn.JoinTable(1)({mAll,mMS})
end
if string.find(opt.dataset,'DS') then
    mAll = nn.JoinTable(1)({mAll,mDS})
end
if string.find(opt.dataset,'RS') then
    mAll = nn.JoinTable(1)({mAll,mRS})
end
if string.find(opt.dataset,'TF') then
    mAll = nn.JoinTable(1)({mAll,mTF})
end

mAll = nn.View(1,numModality,1,500)(mAll)
pretModel = nn.gModule({allInp},{mAll})
pretModel:cuda()
