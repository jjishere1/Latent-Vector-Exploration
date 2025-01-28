import os
import numpy as np
from vtk import *
from vtk.util import numpy_support
import matplotlib.pyplot as plt



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


##################
convert arrays

numpy_array = numpy_support.vtk_to_numpy(vtk_array)
vtk_array = numpy_support.numpy_to_vtk(numpy_array)


data = read_vti(filename)
array = data.GetPointData().GetArray(0)

dim = data.GetDimensions()
spacing = data.GetSpacing()
origin = data.GetOrigin()



new_data = createVtkImageData(origin, dimensions, spacing)
new_data.GetPointData().AddArray(vtk_array)

writeVti(new_data, 'out.vti')




