# Interlayer Dislocations
Calculate the structure or energy of interlayer dislocations in bilayer graphene using a method developed by Murray Daw.

Cite as:


# Prerequisites

The provided code is written in python and C++. For python, it assumes that you have the numpy and scipy packages installed. For C++ it assumes you have the eigen module installed.

python = 2.7.16; numpy=1.16.6; scipy0.13.0b1;

[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) = 3.3.90

    make sure Eigen is in your path variable

# Installation

Compile the C++ with the following command.

`python setup_ID_DAW.py build_ext --inplace`

Note this is for python 2.7 . You may need to specify a specific compiler flag as well (e.g. CC=g++).

This will give you a module named '_IDDAW' that is referenced from python.

# Testing

A few sample calculations are given in the 'Daw_interlayer.py' file.

# Usage

The code works by inputing the material parameters (C, A), supercell dimensions (a1, a2, a3), and the dislocation geometry (array of numpy arrays).
A dislocation is specified as a 3x3 numpy array [location,line_direction,burgers_vector].
