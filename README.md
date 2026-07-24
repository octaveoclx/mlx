# MLX OpenCL Backend – Progress & Features

This document summarizes the current state of the MLX OpenCL backend, highlighting key features, implemented primitives, and the overall progress. The backend aims to provide a complete, high‑performance OpenCL implementation of MLX’s core operations, with a focus on portability and distributed training.

---
2026.7.24

96% tests passed, 8 tests failed out of 224

Total Test time (real) =  72.42 sec

The following tests FAILED:
	172 - test conv1d (SEGFAULT)
	180 - test conv_transpose2d with output_padding (Failed)
	181 - test conv_transpose3d with output_padding (Failed)
	182 - test fp8 conversion (Failed)
	194 - test categorical (SEGFAULT)
	198 - test access stream in other thread (Failed)
	223 - tests (SEGFAULT)
	224 - teardown (Not Run)


2026.7.20

82% tests passed, 42 tests failed out of 239

Total Test time (real) =  79.48 sec

The following tests FAILED:

96 - test fftshift and ifftshift (Failed)  
116 - test single array serialization (Failed)  
118 - test reshape (Failed)  
123 - test slice update (Failed)  
124 - test slice update add (Failed)  
127 - test split (Failed)  
129 - test transpose (Failed)  
130 - test comparison ops (Failed)  
135 - test reduction ops (SEGFAULT)  
137 - test arithmetic unary ops (Failed)  
138 - test error functions (Failed)  
139 - test arithmetic binary ops (Failed)  
140 - test broadcast (Failed)  
142 - test take (Subprocess aborted)  
144 - test put along axis (Failed)  
145 - test scatter (Subprocess aborted)  
149 - test scatter types (Failed)  
151 - test as_strided op (SEGFAULT)  
169 - test quantize dequantize (Failed)  
170 - test repeat (Failed)  
171 - tile (Failed)  
172 - tensordot (Failed)  
175 - test divmod (Failed)  
187 - test conv1d (SEGFAULT)  
190 - test view (Failed)  
191 - test roll (Failed)  
195 - test conv_transpose2d with output_padding (Failed)  
196 - test conv_transpose3d with output_padding (Failed)  
197 - test fp8 conversion (Failed)  
198 - test max min with nan (Failed)  
200 - test global rng (Failed)  
201 - test random split (Failed)  
202 - test random bits (Subprocess aborted)  
206 - test random randint (Failed)  
207 - test random bernoulli (Failed)  
208 - Test truncated normal (Failed)  
209 - test categorical (Failed)  
213 - test access stream in other thread (Failed)  
224 - test simple vmap (Failed)  
232 - test vmap scatter (Failed)  
238 - tests (SEGFAULT)  
239 - teardown (Not Run)

2026.7.6
# Convolution Tests

The test suite verifies the correctness of convolution operations on both CPU and GPU backends. The following parameters are covered:

- **Symmetric padding** (per spatial dimension)
- **Stride** (1D, 2D, and 3D with various values)
- **Dilation** (kernel dilation)
- **Flip** (convolution with flipped kernel)
- **Groups** (grouped convolution)
- **Edge cases** (kernel larger than input, zero-size output)

Each test generates random input and weight tensors, computes the convolution on CPU (reference) and GPU (OpenCL), and compares the results with a tolerance. All tests passed on the tested hardware.

Additionally, performance benchmarks are provided separately to measure speedups.


2026.7.5
- Huge no. of time tick-tocks have been spent on fixing old clMagma problems offering on [AnyMagma](https://github.com/octaveoclx/AnyMagma) to pass all single card tests.
- Implemeted Scan (Prefix Sum) Primitives.

## ✅ Latest Milestone: Scan (Prefix Sum) Primitives – All Tests Pass

The OpenCL backend now fully supports **prefix scan (cumulative reduction)** operations. The complete test suite has been validated against the CPU reference implementation, confirming both **correctness** and **performance** across many workloads.

### Operations Supported

| Operation              | Variants                                     |
|------------------------|----------------------------------------------|
| **Cumulative Sum**     | inclusive, exclusive, forward, reverse      |
| **Cumulative Product** | inclusive, exclusive, forward, reverse      |
| **Cumulative Maximum** | forward                                      |
| **Cumulative Minimum** | forward                                      |

All operations are accelerated on the OpenCL device (GPU) and produce bit‑exact results compared to the CPU fallback.

### Test Coverage

- **1D Tensors**  
  - Small arrays (10 elements) – verify basic per‑workgroup logic.  
  - Large arrays (2048 elements) – exercise cross‑workgroup reduction and the two‑stage reduction path.

- **Multi‑Dimensional Tensors**  
  - 2D arrays (3×4, 64×64) with scanning along axis=0 and axis=1.  
  - 3D arrays (16×32×16) with scanning along axis=1.

- **Data Types**  
  - `float32` – default floating‑point type.  
  - `int32` – integer scan.

- **Edge Cases**  
  - Empty arrays (zero‑size tensors) are handled cleanly without crashes or errors.

### Validation Strategy

- **Reference**: All GPU outputs are compared against the CPU implementation of the same primitive using a tolerance of `1e‑5` for floating‑point types.  
- **Reproducibility**: Random input data is generated with a fixed seed, ensuring deterministic results.  
- **Performance**: Large‑array tests confirm that the kernel correctly decomposes work across multiple work‑groups when the scan dimension exceeds a single work‑group’s capacity.

### Practical Applications

These scan primitives are essential building blocks for:

- Prefix sums in attention mechanisms and cumulative loss calculations.  
- Cumulative products in normalization layers (e.g., layer normalization, RMSNorm).  
- Cumulative min/max for gradient clipping, boundary checks, or running statistics.  
- Parallel prefix algorithms used in sorting, stream compaction, and more.

**Status:** ✅ All tests PASS.

---

## 📅 Detailed Progress Log (Reverse Chronological)

### 2026-06-27
- **Distributed send & recv** via [PoCL](https://github.com/pocl/pocl), tested on two machines for P2P copy using pocl‑remote.
- Fixed old clMAGMA problems (via [AnyMagma](https://github.com/octaveoclx/AnyMagma)) to make inverse and LU decomposition ready for MLX‑OpenCL.

### 2026-06-21
- **1D FFT** single‑dimension/axis support via [VkFFT](https://github.com/DTolm/VkFFT).

### 2026-06-20
- Integrated [AnyMagma](https://github.com/octaveoclx/AnyMagma) (clMAGMA) for matrix inversion.  
  Initially encountered issues with small‑matrix inversion; resolved by using underlayered functions instead of `getri`.  
  As of 6.22, this clMAGMA problem has been fixed with an update in AnyMagma.

### 2026-06-19 – Matrix Multiplication (`matmul`)
- **Primary accelerator:** [CLBlast](https://github.com/CNugteren/CLBlast) – an optimized OpenCL BLAS library.
- **Supported data types:**
  - `float32` – fully accelerated (single and batched GEMM).
  - `float64` – attempted if device supports double precision (otherwise CPU fallback).
  - `float16` – attempted via CLBlast’s `Hgemm`; falls back to CPU if not available.
  - `complex64` – support included CGEMM.
- **Features:**
  - Batched matrix multiplication (3D+ tensors) via strided batched GEMM.
  - Automatic handling of non‑contiguous inputs/outputs (via staging).
  - Transparent fallback to CPU if CLBlast is unavailable or fails.

### 2026-06-18 – Scatter Operations

The following scatter operations are fully implemented and validated:

| Operation     | Description                     |
|---------------|---------------------------------|
| `scatter`     | Replace (single‑ and multi‑axis) |
| `scatter_add` | Accumulate by addition          |
| `scatter_prod`| Accumulate by multiplication    |
| `scatter_max` | Accumulate by maximum           |
| `scatter_min` | Accumulate by minimum           |

**Key Features:**
- Index types: `int32` and `int64`.
- Data types: `float16`, `float32`.
- Negative axes are automatically normalized.
- Empty tensors and out‑of‑bounds indices are handled gracefully.

All operations are tested with stride‑aware comparison logic, ensuring correctness for both contiguous and non‑contiguous layouts.

### Shape & View Operations

| Operation | Implementation Status |
|-----------|-----------------------|
| `Reshape`, `Flatten`, `Unflatten`, `ExpandDims`, `Squeeze`, `BroadcastAxes` | ✅ GPU (zero‑copy shared buffer) |
| `Transpose` | ✅ GPU (zero‑copy) |
| `View` | ✅ GPU (shared buffer or CPU fallback) |

### Slicing & Concatenation

| Operation              | Implementation Status |
|------------------------|-----------------------|
| `Slice`                | ✅ GPU (dedicated kernel `slice_unary`, all tests passed) |
| `DynamicSlice`         | ✅ GPU (via `copy_gpu_inplace`) |
| `SliceUpdate`          | ✅ GPU (dedicated kernel `slice_update_unary`, with row‑contiguous check) |
| `DynamicSliceUpdate`   | ✅ GPU (dedicated kernel `slice_update_unary`) |
| `Concatenate`          | ✅ GPU (via `copy_gpu_inplace`) |

### 2026-06-17 – Unary Operations

The following unary primitives have dedicated OpenCL kernel implementations and pass all unit tests.

| Category                  | Operations                                                               |
|---------------------------|--------------------------------------------------------------------------|
| **Basic Math**            | `Abs`, `Negative`, `Square`, `Ceil`, `Floor`, `Round`, `Sign`           |
| **Exponential & Log**     | `Exp`, `Expm1`, `Log`, `Log1p` (supports natural, base‑2, base‑10)      |
| **Trigonometric**         | `Sin`, `Cos`, `Tan`, `ArcSin`, `ArcCos`, `ArcTan`                       |
| **Hyperbolic & Inverse**  | `Sinh`, `Cosh`, `Tanh`, `ArcSinh`, `ArcCosh`, `ArcTanh`                 |
| **Power & Root**          | `Sqrt`, `Rsqrt` (controlled via `recip` parameter)                     |
| **Special Functions**     | `Erf`, `ErfInv`                                                         |
| **Logical & Bitwise**     | `LogicalNot`, `BitwiseInvert`                                           |
| **Complex**               | `Real`, `Imag`, `Conjugate`                                             |
| **Activation**            | `Sigmoid` (dedicated primitive)                                         |

**NN Functions That Do NOT Require Separate Implementation**  
These are composed from already‑supported basic operations:

- `ReLU` → `maximum(0, x)`
- `Leaky ReLU` → `maximum(negative_slope * x, x)`
- `PReLU` → `max(0, x) + a * min(0, x)`
- `Swish / SiLU` → `x * sigmoid(x)`
- `GELU` → `0.5 * x * (1 + erf(x / sqrt(2)))`
- `Softmax` → `exp(x) / sum(exp(x))`
- `LogSoftmax` → `log(softmax(x))`
- `ELU`, `SELU` – use `exp`, `where`, and arithmetic.

Since their building blocks are already GPU‑accelerated, these high‑level functions automatically run on the OpenCL backend without extra kernel development.

### 2026-06-16
- **Copy improvements:**  
  - `copy_unary` now correctly handles arbitrary strides, enabling proper GPU‑side copies for 3D transposed data.
- **Transpose** and **Reshape** are now GPU‑accelerated using zero‑copy views (`transpose_in_eval`, `reshape_in_eval`), eliminating segmentation faults.
- **reshape_gpu** now only uses zero‑copy when the input is row‑contiguous; otherwise forces an explicit copy to produce a truly contiguous output.

### 2026-06-13
- Added **erfinv** (from Prof. Mike Giles’s code) and **FP64** support.
- Full set of unary operations now coded.

### 2026-06-11
- **FP16 support in CLBlast** for Apple Silicon and NVIDIA GPUs (with help from an ICD wrapper).
- **bf16** simulated via float, with promote/demote macros in kernels.
- Support for **UMA** (Apple Silicon, Intel Xe laptop GPU+CPU) and standard copy‑buffer behavior for discrete GPUs.

### 2026-06-09
- Flexible type support using the same kernel differentiated by the `TYPE` macro.
- Direct binary add and broadcast add working for all supported types.

### 2026-06-08
- Aligned with [Vulkan backend (2026.3.5)](https://github.com/NripeshN/mlx/commit/09371e55508518caadcc05f1aa2ea3d2225fdcac).  
  Core GPU kernel dispatch functions (binary, unary, reduce, softmax, scan, etc.) are placeholder implementations; actual OpenCL kernel code is being written.

### 2026-06-07
- Successfully created and built a basic OpenCL framework aligned with [Vulkan backend (2026.3.4)](https://github.com/NripeshN/mlx/commit/d64d1ffb7479cfa46b7cb8525f6a46704ab25498).

---

## Why MLX + OpenCL Is a Promising Direction

MLX has significant untapped potential when combined with OpenCL. Here’s why the time is right to start this work.

### 1. MLX’s Architecture Is Naturally Suited for OpenCL
MLX has a clean, layered design with a well‑defined backend abstraction (`Primitive::eval_gpu`). Existing Metal and Vulkan backends demonstrate how to implement compute kernels without heavy runtime dependencies. Adding an OpenCL backend fits directly into this model – reusing the same 100–200 core primitives.

### 2. A Manageable Number of Primitives Makes Collaboration Feasible
Unlike PyTorch (which has 2000+ operators), MLX requires only about 100–200 kernel primitives to reach full functionality. This small scale means a small team (or even a dedicated individual) can realistically implement all required GPU kernels for OpenCL.

### 3. PoCL‑Remote Enables Distributed Training – Like NCCL but Open
[PoCL‑remote](http://portablecl.org/docs/html/remote.html) allows OpenCL devices across a network to appear as local devices. By building a collective communication layer on top (AllReduce, Broadcast, etc.), we can create an **NCCL‑like distributed training framework** that works on any hardware supporting OpenCL. This is especially valuable in the era of big data, where cost‑effective consumer GPUs or accelerators can be interconnected via standard Ethernet.

### 4. Lower the Risk and Shorten the Development Curve

The widespread success of CUDA in accelerating machine learning workloads, together with the recent emergence of a Vulkan backend for MLX in just the past few months, provides an important and practical reference for this work.

We leverage several mature OpenCL‑based libraries:

- **CLBlast** – Optimised BLAS, heavily tuned for matrix multiplication.
- **vkFFT** – Provides an OpenCL interface for FFT; valuable for spectral operations.
- **AnySparse** – Our revived version of clSparse, offering efficient sparse solvers.
- **AnyMagma** – Our revived version of clMAGMA, useful for matrix decompositions and dense linear algebra.
- **AnyArray** – Derived from Octave’s ocl; serves as our version of a GPU array, similar to MATLAB’s gpuArray.
- **PoCL** – Experience configuring PoCL for dual devices on Apple Silicon and using PoCL‑remote for cluster setups.

### 5. Why MLX Reduces the Number of Operators – from DeepSeek

In traditional frameworks like PyTorch’s ATen, covering various combinations (e.g., the gradient of `sin(cos(x))`, batched `sin`, or a fused `sin+cos+exp` kernel) often requires:

- Explicitly implementing forward operators: `Sin`, `Cos`, `Mul`, `Exp`, etc.
- Explicitly implementing backward operators: `SinBackward`, `CosBackward`, `MulBackward`, etc.
- Explicitly implementing batched versions: `BatchSin`, `BatchCos` (or relying on broadcasting, which often still requires separate optimizations).
- Manually writing fused kernels like `FusedSinCosExpKernel` and their corresponding backward pass.

MLX, in contrast, implements only the most basic forward kernels (e.g., `sin`, `cos`, `mul`, `exp`) along with their VJP (vector-Jacobian product) rules. Then, through three powerful function transforms:

- `grad` → automatically generates the reverse pass for any arbitrarily complex function.
- `vmap` → automatically generates batched versions.
- `compile` → automatically generates fused kernels.

The synergy of these three transforms allows MLX to cover the same functional space that would require hundreds or even thousands of operators in frameworks like PyTorch, using only a few dozen basic primitives.

### Summary

- ✅ MLX’s simple backend interface lowers the porting effort.
- ✅ A small set of primitives keeps the task tractable.
- ✅ PoCL‑remote offers a path to open, multi‑vendor distributed training.

This work brings MLX one step closer to becoming a truly portable, high‑performance machine learning framework, ready to run on a wide variety of hardware from laptops to multi‑node clusters.
```

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
