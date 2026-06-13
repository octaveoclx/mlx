# Progress
2026.6.13
OpenCL conversion of Prof. Mike Giles's work on [erfinv](https://people.maths.ox.ac.uk/gilesm/codes/erfinv/).
Add FP64 support.
Full Unary coded.

2026.6.11

Add support of FP16 in CLBLAST for [Apple Silicon and NVIDIA GPUs](https://github.com/CNugteren/CLBlast/commit/f78f6dd0edd5f24441f61bfade262e8a0684ce70). 
[Discussion](https://github.com/CNugteren/CLBlast/issues/667) 
[Discussion2](https://forums.developer.nvidia.com/t/gtx-1660-super-tu116-not-exposing-fp16-on-driver-580-94-16/359199/6])

Add FP16 in the Apple Silicon with help of [ICD warpper](https://github.com/octaveoclx/ocl_icd_wrapper/tree/cl_khr_fp16).

Support direct binary add and broadcast add.

Add promote and demote to kernels to supoort bf16 - float simulation, FP16 - float simulation if does not work, FP32, F64, u/intXX.

UMA enabled with Apple silicon, Intel Xe laptop GPU+CPU, and normal copy buffer behavior for DGPUs.

2026.6.9
Add flexible type support using the same kernel differentiaing by TYPE macro.
```bash
(base) jc@U1:~/Downloads/mlx-feat-vulkan/build$ ./test_add
OpenCL is available.
Device name: Intel(R) Iris(R) Xe Graphics
Default device: gpu
Before eval, c data type: float32
eval_binary_opencl_or_cpu called for add
try_eval_binary_op_opencl called for add
Result: array([5, 7, 9], dtype=float32)
(base) jc@U1:~/Downloads/mlx-feat-vulkan/build$
g++ -std=c++20 -o test_add_fp16 ../test_add_fp16.cpp -I.. -L. -lmlx -lOpenCL -lopenblas -llapack -lgfortran -lpthread
./test_add_fp16
OpenCL device: Intel(R) Iris(R) Xe Graphics
eval_binary_opencl_or_cpu called for add
try_eval_binary_op_opencl called for add
float16 addition result: array([5, 7, 9], dtype=float16)
Expected: [5, 7, 9], got: [5, 7, 9]
(base) jc@U1:~/Downloads/mlx-feat-vulkan/build$ 
```

2026.6.8: align with the work of [vulkan 2026.3.5](https://github.com/NripeshN/mlx/commit/09371e55508518caadcc05f1aa2ea3d2225fdcac). Compared to the Vulkan backend, the OpenCL backend's core GPU kernel dispatch functions (such as binary, unary, reduce, softmax, scan, etc.) are still placeholder implementations that throw exceptions, and no real OpenCL kernel code has been written yet.

2026.6.7: sucessfully create and build a basic OpenCL framework to align with the work of [vulkan 2026.3.4](https://github.com/NripeshN/mlx/commit/d64d1ffb7479cfa46b7cb8525f6a46704ab25498)


## Why MLX + OpenCL is a promising direction

MLX has significant untapped potential when combined with OpenCL. Here’s why the time is right to start this work.

### 1. MLX’s architecture is naturally suited for OpenCL
MLX has a clean, layered design with a well-defined backend abstraction (`Primitive::eval_gpu`). Existing Metal and Vulkan backends already demonstrate how to implement compute kernels without heavy runtime dependencies. Adding an OpenCL backend fits directly into this model – reusing the same 100–200 core primitives.

### 2. A manageable number of primitives makes collaboration feasible
Unlike PyTorch (which has 2000+ operators), MLX requires only about 100–200 kernel primitives to reach full functionality. This small scale means a small team (or even a dedicated individual) can realistically implement all required GPU kernels for OpenCL, without needing a massive contributor base.

### 3. PoCL‑remote enables distributed training – like NCCL but open
[PoCL‑remote](http://portablecl.org/docs/html/remote.html) allows OpenCL devices across a network to appear as local devices. By building a collective communication layer on top (AllReduce, Broadcast, etc.), we can create an **NCCL‑like distributed training framework** that works on any hardware supporting OpenCL. This is especially valuable in the era of big data, where cost‑effective consumer GPUs or accelerators can be interconnected via standard Ethernet – network speed becomes the primary bottleneck, but the approach is still practical for many workloads.

### 4. Lower the risk and shorten the development curve

The widespread success of CUDA in accelerating machine learning workloads, together with the recent emergence of a Vulkan backend for MLX in just the past few months, provides an important and practical reference for this work. 

clBLAST – Heavily involved in tuning; its optimisations (especially for matrix multiplication) can be adapted for MLX’s core primitives.

vkFFT – Provides an OpenCL interface for FFT; valuable for spectral operations in MLX.

AnySparse – Our revived version of clSparse, offering efficient sparse problem solvers.

AnyMagma – Our revived version of clMAGMA, useful for matrix decompositions and dense linear algebra.

AnyArray – Derived from Octave’s ocl; serves as our version of a GPU array, similar to MATLAB’s gpuArray.

PoCL – We have experience configuring PoCL for dual devices on Apple Silicon and using PoCL‑remote for cluster setups.


### Summary
- ✅ MLX’s simple backend interface lowers the porting effort.
- ✅ A small set of primitives keeps the task tractable.
- ✅ PoCL‑remote offers a path to open, multi‑vendor distributed training.

If you are interested in contributing to an OpenCL backend for MLX, let’s connect!

From Prof. Jinchuan Tang



# MLX

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://ml-explore.github.io/mlx/build/html/index.html) |
[**Examples**](#examples)

[![CircleCI](https://circleci.com/gh/ml-explore/mlx.svg?style=svg)](https://circleci.com/gh/ml-explore/mlx)

MLX is an array framework for machine learning on Apple silicon,
brought to you by Apple machine learning research.

Some key features of MLX include:

- **Familiar APIs**: MLX has a Python API that closely follows NumPy. MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API. MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

- **Composable function transformations**: MLX supports composable function
  transformations for automatic differentiation, automatic vectorization,
  and computation graph optimization.

- **Lazy computation**: Computations in MLX are lazy. Arrays are only
  materialized when needed.

- **Dynamic graph construction**: Computation graphs in MLX are constructed
  dynamically. Changing the shapes of function arguments does not trigger
  slow compilations, and debugging is simple and intuitive.

- **Multi-device**: Operations can run on any of the supported devices
  (currently the CPU and the GPU).

- **Unified memory**: A notable difference from MLX and other frameworks
  is the *unified memory model*. Arrays in MLX live in shared memory.
  Operations on MLX arrays can be performed on any of the supported
  device types without transferring data.

MLX is designed by machine learning researchers for machine learning
researchers. The framework is intended to be user-friendly, but still efficient
to train and deploy models. The design of the framework itself is also
conceptually simple. We intend to make it easy for researchers to extend and
improve MLX with the goal of quickly exploring new ideas.

The design of MLX is inspired by frameworks like
[NumPy](https://numpy.org/doc/stable/index.html),
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/).

## Examples

The [MLX examples repo](https://github.com/ml-explore/mlx-examples) has a
variety of examples, including:

- [Transformer language model](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) training.
- Large-scale text generation with
  [LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) and
  finetuning with [LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora).
- Generating images with [Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
- Speech recognition with [OpenAI's Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

## Quickstart

See the [quick start
guide](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)
in the documentation.

## Installation

MLX is available on [PyPI](https://pypi.org/project/mlx/). To install MLX on
macOS, run:

```bash
pip install mlx
```

To install the CUDA backend on Linux, run:

```bash
pip install mlx[cuda]
```

To install a CPU-only Linux package, run:

```bash
pip install mlx[cpu]
```

Checkout the
[documentation](https://ml-explore.github.io/mlx/build/html/install.html#)
for more information on building the C++ and Python APIs from source.

## Contributing

Check out the [contribution guidelines](https://github.com/ml-explore/mlx/tree/main/CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

We are grateful for all of [our
contributors](https://github.com/ml-explore/mlx/tree/main/ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX and wish to be acknowledged, please add your name to the list in your
pull request.

## Citing MLX

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find
MLX useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```
