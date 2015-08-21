
--[[

This file makes plots similar to those in the blog post.
Requires fblualib.

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-valid_data_path','file with validation text')
cmd:option('-layer_idx', 'layer index')
cmd:option('-neuron_idx','neuron index within a given layer; we will monitor its state')
cmd:option('-ntop_predictions',5, 'top character predictions')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state = {}
print(checkpoint.opt.num_layers)
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):float()
    if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end

-- validation data
local f = io.open(opt.valid_data_path, "r")
text = f:read("*all")
f:close()

--
seq_len = #text
state_size = #current_state
seq_states = {} 
predictions = {}

for i = 1,seq_len do
    local c = text:sub(i,i)
    prev_char = torch.Tensor{vocab[c]}
    
    if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
  
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    predictions = lst[#lst] -- last element holds the log probabilities
    
    local current_hidden_state = {}
    for i=2,state_size,2 do 
      table.insert(current_hidden_state, lst[i]) 
    end
    
    
    local _, max_char_ = predictions:max(2)
    max_char = max_char_:resize(1)
    
    local p, idx = torch.sort(predictions, true)
    p = p:squeeze()
    
    local max_predictions = {}
    for i=1,opt.ntop_predictions do
      local char = ivocab[idx[{{},i}]:squeeze()]
      local p_char = p[i]  
      max_predictions[char] = p_char 
    end
   
    seq_states[tostring(i).."_input_char"] = c
    seq_states[tostring(i).."_output"] = max_predictions 
    seq_states[tostring(i).."_state"]= {}
    for j = 1, #current_hidden_state do
        seq_states[tostring(i).."_state"][j] = current_hidden_state[j]:squeeze():clone()
    end
end

 -- plots
py = require('fb.python')
py.exec([=[ 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

seq_len = int(seq_len)
ntop_predictions = int(ntop_predictions)
layer_idx = int(layer_idx)
neuron_idx = int(neuron_idx)
max_val = 10
plt_seq_len =50
filename = os.path.basename(os.path.splitext(filename)[0])

data_char = np.zeros((1+ntop_predictions,seq_len), dtype='<U1')
data = np.zeros((1+ntop_predictions,seq_len))

for j in range(int(seq_len)):

    sd = sorted(seq_states[str(j+1)+'_output'].items(),key=lambda x: x[1], reverse=True)    
    data[1:,j] = [np.exp(x[1]) for x in sd]    
    data_char [1:,j] = [x[0] for x in sd]
    
    data_char [0,j] = seq_states[str(j+1)+'_input_char']
    data[0,j] = seq_states[str(j+1)+'_state'][layer_idx][neuron_idx]

# rescale hidden states to [0,max_val] interval and predictions to [-max_val,0]
data[0,:]= (data[0,:] - data[0,:].min()) / (data[0,:].max() - data[0,:].min()) * max_val
data[1:,:]= -1.0 *(data[1:,:] - data[1:,:].min()) / (data[1:,:].max() - data[1:,:].min()) * max_val

nfig = int(seq_len/(plt_seq_len*3))
seq_idx = 0
for fig_idx in np.arange(nfig):
    plt.figure()
        
    for plt_idx in np.arange(3):
      plt.subplot(3,1,plt_idx+1)
      plt.pcolor(data[:,seq_idx:seq_idx + plt_seq_len], 
                 cmap=cm.RdBu, vmin=-max_val, vmax=max_val)
      
      for y in range(ntop_predictions+1):
          for x in range(plt_seq_len):
              plt.text(x + 0.5, y + 0.5, str(data_char[y, x+seq_idx]),
                       weight ='medium',
                       horizontalalignment='center', verticalalignment='center')
      
      seq_idx += plt_seq_len
      
      ax = plt.gca() 
      ax.invert_yaxis()
      ax.set_yticks(np.arange(ntop_predictions+1))
      ax.set_xticks(np.arange(50))
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      plt.grid(True)      
      
    plt.show()
    #plt.savefig('%s%s_l%su%s.png'% (filename, fig_idx, layer_idx, neuron_idx)) 

]=], {seq_states = seq_states, seq_len=seq_len,
      ntop_predictions=opt.ntop_predictions,
      layer_idx = opt.layer_idx, neuron_idx=opt.neuron_idx, 
      filename = opt.valid_data_path})