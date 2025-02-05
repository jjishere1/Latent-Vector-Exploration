import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from vtkmodules.util import numpy_support
import vtk
from train import (train_model, custom_loss, evaluate_model)
from model import Conv3DAutoencoder
from utils import (read_vti, get_numpy_array_from_vtk_image_data, writeVti,
                   createVtkImageData, compute_gradient_magnitude, tensor_to_vtk_image_data,
                   compute_PSNR)
from datasets import (BlockDataset, divide_into_blocks, pad_blocks, unpad_block, reassemble_blocks)

def parse_dims(s):
    return tuple(map(int, s.split(',')))

def main():
    parser = argparse.ArgumentParser(description="3D Conv Autoencoder Training Script")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--block_dims", type=str, default="5,5,5", help="Block dimensions (d,h,w), e.g. 5,5,5")
    parser.add_argument("--padded_block_dims", type=str, default="9,9,9", help="Padded block dimensions (d,h,w), e.g. 9,9,9")
    parser.add_argument("--use_padding", action='store_true', help="Use padding (if not set, use divide_into_blocks)")
    parser.add_argument("--output_vti", type=str, default="out.vti", help="Name for output VTI file")
    parser.add_argument("--loss_file", type=str, default="training_loss.txt", help="File to save training loss")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    block_dims = parse_dims(args.block_dims)           # e.g. (5,5,5)
    padded_dims = parse_dims(args.padded_block_dims)     # e.g. (9,9,9)
    use_padding = args.use_padding
    output_vti_filename = args.output_vti
    loss_file = args.loss_file

    input_vti = os.path.join("datasets", "data.vti")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)


    vti_data = read_vti(input_vti)
    try:
        data_np = get_numpy_array_from_vtk_image_data(vti_data)
        print("Data shape from VTI:", data_np.shape)
    except ValueError as e:
        print(e)
        return

    data_np = data_np.astype(np.float32)
    data_np = 2 * ((data_np - data_np.min()) / (data_np.max() - data_np.min())) - 1

    input_tensor = torch.from_numpy(data_np).unsqueeze(0)

    pad_value = 0  

    if use_padding:
        blocks = pad_blocks(input_tensor, block_dims, padded_dims, pad_value)
    else:
        blocks = divide_into_blocks(input_tensor, block_dims)

    dataset = BlockDataset(blocks)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv3DAutoencoder().to(device)
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = train_model(model, dataloader, device, num_epochs, 0.001)
  
    loss_path = os.path.join(output_dir, loss_file)
    with open(loss_path, "w") as f:
        for i, loss_val in enumerate(epoch_losses):
            f.write(f"Epoch {i+1}: {loss_val}\n")
    print(f"Training loss saved to {loss_path}")

    evaluate_model(model, blocks, device)

    if use_padding:
        reconstructed_blocks = [unpad_block(b, block_dims, padded_dims) for b in reconstructed_blocks]

    reassembled_data = reassemble_blocks(reconstructed_blocks, input_tensor.shape, block_dims)
    
    # Compute final MSE loss 
    original_tensor = input_tensor.to(device)
    reassembled_tensor = torch.tensor(reassembled_data, dtype=torch.float).to(device)
    final_loss = mse_criterion(reassembled_tensor , original_tensor)
    print(f'Final MSE Loss on Entire Data: {final_loss.item():.6f}')

    # Compute PSNR 
    reconstructed_np = reassembled_tensor.squeeze().cpu().numpy()
    original_np = original_tensor.squeeze().cpu().numpy()
    psnr_score = compute_PSNR(original_np, reconstructed_np)
    print("PSNR Score:", psnr_score)

    # Save the reconstructed volume as a VTI file
    flat_data = reconstructed_np.flatten(order='F')
    vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
    dims = vti_data.GetDimensions()
    spacing = vti_data.GetSpacing()
    origin = vti_data.GetOrigin()
    new_vtk_data = createVtkImageData(origin, dims, spacing)
    new_vtk_data.GetPointData().AddArray(vtk_array)
    output_vti_path = os.path.join(output_dir, output_vti_filename)
    writeVti(new_vtk_data, output_vti_path)
    print("Reconstructed VTI saved to", output_vti_path)

if __name__ == "__main__":
    main()