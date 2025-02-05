import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = [torch.tensor(block, dtype=torch.float) for block in blocks]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def pad_blocks(data, block_shape, padded_shape, pad_value=0):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float)
    if data.ndim == 3:
        data = data.unsqueeze(0)
    pad_h = padded_shape[0] // 2
    pad_w = padded_shape[1] // 2
    pad_d = padded_shape[2] // 2
    padded_data = F.pad(data, (pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=pad_value)
    _, H, W, D = data.shape
    n_blocks_h = H // block_shape[0]
    n_blocks_w = W // block_shape[1]
    n_blocks_d = D // block_shape[2]
    blocks_list = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            for k in range(n_blocks_d):
                center_h = i * block_shape[0] + (block_shape[0] // 2)
                center_w = j * block_shape[1] + (block_shape[1] // 2)
                center_d = k * block_shape[2] + (block_shape[2] // 2)
                c_h = center_h + pad_h
                c_w = center_w + pad_w
                c_d = center_d + pad_d
                block = padded_data[:, 
                    c_h - pad_h : c_h + pad_h + 1,
                    c_w - pad_w : c_w + pad_w + 1,
                    c_d - pad_d : c_d + pad_d + 1
                ]
                blocks_list.append(block)
    return blocks_list

def unpad_block(padded_block, block_dims, padded_shape):
    start_h = padded_shape[0] // 2 - block_dims[0] // 2
    start_w = padded_shape[1] // 2 - block_dims[1] // 2
    start_d = padded_shape[2] // 2 - block_dims[2] // 2
    return padded_block[:, start_h:start_h + block_dims[0],
                           start_w:start_w + block_dims[1],
                           start_d:start_d + block_dims[2]]

def reassemble_blocks(blocks, original_shape, block_dims, padded_shape):
    reassembled_data = np.zeros(original_shape)
    index = 0
    for h in range(0, original_shape[1], block_dims[0]):
        for w in range(0, original_shape[2], block_dims[1]):
            for d in range(0, original_shape[3], block_dims[2]):
                block = blocks[index]
                unpadded = unpad_block(block, block_dims, padded_shape)
                block_data = unpadded.numpy()
                reassembled_data[:, h:h + block_dims[0],
                                  w:w + block_dims[1],
                                  d:d + block_dims[2]] = block_data
                index += 1
    return reassembled_data

def divide_into_blocks(data, block_dims): # make blocks 5x5x5 

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float)

    blocks = []
    height_step, width_step, depth_step = block_dims

    for h in range(0, data.shape[1], height_step): 
        for w in range(0, data.shape[2], width_step):  
            for d in range(0, data.shape[3], depth_step): 
                block = data[:, h:h + height_step, w:w + width_step, d:d + depth_step]
                blocks.append(block)

    return blocks
