from setuptools import setup, Extension
from torch.utils import cpp_extension
#BuildExtension,CppExtension,CUDAExtension
setup(name='meshing_occupancy',
    ext_modules=[cpp_extension.CUDAExtension('meshing_occupancy', ['meshing_occupancy.cpp','meshing_occupancy_device.cu'])],
    # ext_modules=[cpp_extension.CppExtension('meshing_occupancy', ['meshing_occupancy.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})