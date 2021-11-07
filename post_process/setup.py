from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bpd_cpu',
    ext_modules=[
        CUDAExtension('bpd_cpu', [
            'bpd_cpu.cpp',
            'bpd_cpu_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })