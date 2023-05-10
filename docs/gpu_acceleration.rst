GPU Accelerated Optical Calculations
====================================



.. admonition:: Placeholder docs
   This page is a placeholder for more complete documentation to be added later about usage of GPUs for fast optical calculations.



Thanks to team members Kian Milani and Ewan Douglas (University of Arizona), poppy now includes a
high performance option using NVidia GPUs to significantly accelerate optical calculations, in some
cases by ~20x to 80x. 

This implementation seeks to perform all calculations on the GPU until the end of propagation. This
reduces time for calculations as arrays no longer need to be transferred between GPU memory and
standard memory when performing different calculations. It also allows GPU acceleration of the
majority of calculations performed during an optical propagation (i.e. creating models of optical
elements and applying them to wavefronts happens on the GPU, as well as the propagation calculations
from one plane to another.)

An updated implementation using `CuPy <https://docs.cupy.dev/en/stable/overview.html>` replaces
initial earlier support for CUDA using pyculib and numba.cuda. (That initial implementation has been
removed since the CuPy implementation is much better performing.)

Note, because cupy is used as a replacement for numpy at import time, it is a bit tricky to toggle
between GPU and CPU calculations during the same python session. Doing so is advanced usage, and
while it can be useful in some cases for debugging or benchmarking, it's not fully supported or
recommended to try to switch between calculation backends within the same session. 


**What about AMD GPUs?**

There also exists partial/earlier support for OpenCL for AMD GPUs, using the `pyopencl` and `gpyfft`
packages. This provides much less performance gains than the CuPy version, however, since only 
FFTs are performed on-GPU, not other parts of the optical propagation calculations.

**What about Apple Silicon GPUs?**

Poppy does not yet have support for the specialized GPU hardware in Apple Silicon M1/M2 and similar.
For these machines, plain numpy is the best option.

Requirements and Setup
----------------------


Requires NVidia GPU hardware

Requires CuPy > 10.0. Install from https://cupy.dev following the `CuPy installation docs <https://docs.cupy.dev/en/stable/install.html#>`_

Also requires the cupyx GPU-accelerated version of scipy.


Performance Comparisons
-----------------------



Computation comparisons have been performed to illustrate the benefit of this accelerated computing
feature. Below are comparisons of the times required for a PSF to be calculated for varying array
sizes using the MKL FFT option versus the CuPy calculations. The optical systems tested had 5
different surfaces/optics. 

Performances will naturally vary depending on the compute hardware used. The system used for these
comparisons was the University of Arizona’s HPC Puma nodes. The node utilized 32 AMD EPYC 7642 CPUs
and the NVIDIA Tesla V100S GPU.

+-------------------+--------------+------------------------+-------------------------+----------------------+
|  Propagation Type |	Array Size |	MKL Method Times [s] |	CuPy Method Times [s] |	Speed Up Factor      |
+===============+=======+=======+===============+=======+
| Fraunhofer	| 1024	| 0.218	| 0.0261	| 8.35  |
| Fraunhofer	| 2048	| 0.755	| 0.0294	| 25.7  |
| Fraunhofer	| 4096	| 3.36	| 0.0423	| 79.4  |
| Fresnel	| 1024	| 0.714	| 0.0438	| 16.3  |
| Fresnel	| 2048	| 4.16	| 0.0845	| 49.2  |
| Fresnel	| 4096	| 17.5	| 0.225	        | 77.8  |
+---------------+-------+-------+---------------+-------+

