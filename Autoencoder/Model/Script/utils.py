import torch
import torch.nn as nn
import numpy as np
import vtk
from vtk import vtkXMLImageDataReader,  vtkImageGradient, vtkImageMagnitude, vtkXMLImageDataWriter
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.util import numpy_support

def compute_PSNR(arrgt,arr_recon):
    try:
        diff = arrgt - arr_recon
        sqd_max_diff = (np.max(arrgt)-np.min(arrgt))**2
        if(np.mean(diff**2) == 0):
            raise ZeroDivisionError("dividing by zero, cannot calculate psnr")
        snr = 10*np.log10(sqd_max_diff/np.mean(diff**2))
        return snr
    except ZeroDivisionError as err:
        return str(err)

def read_vti(filename):
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def writeVti(data, filename):
    writer = vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Write()


def createVtkImageData(origin, dimensions, spacing):
    localDataset = vtkImageData()
    localDataset.SetOrigin(origin)
    localDataset.SetDimensions(dimensions)
    localDataset.SetSpacing(spacing)
    return localDataset

def compute_gradient_magnitude(vtk_image_data):
    if not isinstance(vtk_image_data, vtk.vtkImageData):
        raise TypeError("compute_gradient_magnitude requires a vtkImageData object")
    
    if vtk_image_data.GetPointData().GetScalars() is None:
        raise ValueError("VTK ImageData has no scalars set. Cannot compute gradients.")

    gradient_filter = vtk.vtkImageGradient()
    gradient_filter.SetInputData(vtk_image_data)
    gradient_filter.SetDimensionality(3)
    gradient_filter.Update()

    magnitude_filter = vtk.vtkImageMagnitude()
    magnitude_filter.SetInputConnection(gradient_filter.GetOutputPort())
    magnitude_filter.Update()

    return magnitude_filter.GetOutput()

def get_numpy_array_from_vtk_image_data(vtk_image_data):
    point_data = vtk_image_data.GetPointData()
    array = point_data.GetScalars()  
    
    if array is None:
        raise ValueError("No scalar array found in vtkImageData.")
    
    numpy_array = numpy_support.vtk_to_numpy(array)
    dims = vtk_image_data.GetDimensions()  
    num_components = array.GetNumberOfComponents()

    expected_size = dims[0] * dims[1] * dims[2] * num_components

    if numpy_array.size != expected_size:
        raise ValueError(f"Shape mismatch! Cannot reshape {numpy_array.size} elements into {dims[2], dims[1], dims[0], num_components}")

    numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0], num_components)  
    return numpy_array.squeeze()  

def tensor_to_vtk_image_data(tensor):
    np_array = tensor.detach().cpu().numpy()

    if np_array.ndim == 4:  
        np_array = np_array[0] 

    vtk_image_data = vtk.vtkImageData()

    dims = np_array.shape  
    vtk_image_data.SetDimensions(dims[2], dims[1], dims[0])  

    flat_array = np_array.flatten(order='F')
    vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image_data.GetPointData().SetScalars(vtk_array)
    
    return vtk_image_data

