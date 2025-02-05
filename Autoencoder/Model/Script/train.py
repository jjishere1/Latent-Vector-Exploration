import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import tensor_to_vtk_image_data, compute_gradient_magnitude, get_numpy_array_from_vtk_image_data, compute_PSNR

def custom_loss(output, target, gradient_target, alpha=0.5):
    mse_criterion = nn.MSELoss()
    reconstruction_loss = mse_criterion(output, target)
    output_vtk = tensor_to_vtk_image_data(output)
    output_gradient_magnitude = compute_gradient_magnitude(output_vtk)
    output_gradient_magnitude_array = get_numpy_array_from_vtk_image_data(output_gradient_magnitude)
    assert output_gradient_magnitude_array.shape == gradient_target.shape, "Shapes must match"
    output_magnitude = np.linalg.norm(output_gradient_magnitude_array)
    gradient_target_magnitude = np.linalg.norm(gradient_target)
    gradient_loss = np.abs(output_magnitude - gradient_target_magnitude)
    total_loss = alpha * reconstruction_loss + (1 - alpha) * gradient_loss
    return total_loss

def train_model(model, dataloader, device, num_epochs,learning_rate):
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_records = []
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            batch_vtk = tensor_to_vtk_image_data(batch)
            batch_gradient_magnitude = compute_gradient_magnitude(batch_vtk)
            batch_gradient_magnitude_array = get_numpy_array_from_vtk_image_data(batch_gradient_magnitude)
            optimizer.zero_grad()
            output = model(batch)
            loss = custom_loss(output, batch, batch_gradient_magnitude_array)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        loss_records.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return loss_records

def evaluate_model(model, blocks, device):
    print("Evaluating model...")
    model.eval()
    reconstructed_blocks = []
    with torch.no_grad():
        for block in blocks:
            block_tensor = block if torch.is_tensor(block) else torch.tensor(block, dtype=torch.float)
            block_tensor = block_tensor.unsqueeze(0).to(device)
            output = model(block_tensor).numpy()
            reconstructed_blocks.append(output.squeeze(0).cpu())
    return reconstructed_blocks

if __name__ == "__main__":
    print("Train module loaded.")
