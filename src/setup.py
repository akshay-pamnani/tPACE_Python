from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11 until it is actually installed,
    so that the `get_include()` method can be invoked."""
    def __str__(self):
        return pybind11.get_include()

eigen_include_path = 'eigen-3.4.0'

# Define extensions
ext_modules = [
    Extension(
        'CPPlwls1d_py',
        sources=['CPPlwls1d_py.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            # Path to Eigen headers
            eigen_include_path,
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
    Extension(
        'trapzRcpp',
        sources=['trapzRcpp.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
        ],
        language='c++',
    ),
    Extension(
        'RcppPseudoApprox',
        sources=['RcppPseudoApprox.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            eigen_include_path,
        ],
        language='c++',
    ),
]

setup(
    name='CPPlwls1d_py',
    version='0.1',
    author='Akshay Pamnani',
    author_email='akshay.iithyd@gmail.com',
    description='Local Weighted Least Squares (LWLS) smoothing function',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
