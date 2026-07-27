# MLX OpenCL Backend – Progress & Features

This document summarizes the current state of the MLX OpenCL backend, highlighting key features, implemented primitives, and the overall progress. The backend aims to provide a complete, high‑performance OpenCL implementation of MLX’s core operations, with a focus on portability and distributed training.

---
2026.7.28

Align to MLX 0.3.20 release, with extra test cases on deep-wise conv and layer normalization.

Next: need to add C++ test cases for FAST primitives.  

	(base) jc@U1:~/Downloads/vdss/mlx-opencwl/mlx-opencl726/mlx-opencl/build/tests$ make tests -j16
	[  0%] Built target cpu_compiled_preamble
	[  0%] Built target mlx_version
	[  1%] Built target gguflib
	[ 57%] Built target opencl_kernels
	[ 57%] Building CXX object CMakeFiles/mlx.dir/mlx/backend/opencl/opencl_convolution.cpp.o
	[ 57%] Linking CXX static library libmlx.a
	[ 92%] Built target mlx
	[ 92%] Linking CXX executable tests
	[OpenCL] Double precision support: NO (fallback to float)
	[OpenCL] Half mode: PROMOTED (float compute, export MLX_OPENCL_NATIVE_HALF=0) on Intel(R) OpenCL Graphics
	[100%] Built target tests
	(base) jc@U1:~/Downloads/vdss/mlx-opencwl/mlx-opencl726/mlx-opencl/build/tests$ make test -j16
	Running tests...
	Test project /home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl726/mlx-opencl/build/tests
	Connected to MAKE jobserver
	        Start   1: test simple allocations
	  1/264 Test   #1: test simple allocations .............................................   Passed    0.06 sec
	        Start   2: test large allocations
	  2/264 Test   #2: test large allocations ..............................................   Passed    0.18 sec
	        Start   3: test cached allocation keeps capacity
	  3/264 Test   #3: test cached allocation keeps capacity ...............................   Passed    0.07 sec
	        Start   4: test clear cache synchronizes cpu streams
	  4/264 Test   #4: test clear cache synchronizes cpu streams ...........................   Passed    0.08 sec
	        Start   5: test arg reduce small
	  5/264 Test   #5: test arg reduce small ...............................................   Passed    0.06 sec
	        Start   6: test arg reduce against cpu
	  6/264 Test   #6: test arg reduce against cpu .........................................   Passed    0.06 sec
	        Start   7: test arg reduce bool
	  7/264 Test   #7: test arg reduce bool ................................................   Passed    0.06 sec
	        Start   8: test arg reduce edge cases
	  8/264 Test   #8: test arg reduce edge cases ..........................................   Passed    0.09 sec
	        Start   9: test arg reduce irregular strides
	  9/264 Test   #9: test arg reduce irregular strides ...................................   Passed    0.06 sec
	        Start  10: test array basics
	 10/264 Test  #10: test array basics ...................................................   Passed    0.10 sec
	        Start  11: test array types
	 11/264 Test  #11: test array types ....................................................   Passed    0.11 sec
	        Start  12: test array metadata
	 12/264 Test  #12: test array metadata .................................................   Passed    0.11 sec
	        Start  13: test array iteration
	 13/264 Test  #13: test array iteration ................................................   Passed    0.07 sec
	        Start  14: test array shared buffer
	 14/264 Test  #14: test array shared buffer ............................................   Passed    0.07 sec
	        Start  15: test make empty array
	 15/264 Test  #15: test make empty array ...............................................   Passed    0.06 sec
	        Start  16: test make array from user buffer
	 16/264 Test  #16: test make array from user buffer ....................................   Passed    0.10 sec
	        Start  17: test negative indexing for shape/strides
	 17/264 Test  #17: test negative indexing for shape/strides ............................   Passed    0.08 sec
	        Start  18: test siblings circular references without eval
	 18/264 Test  #18: test siblings circular references without eval ......................   Passed    0.07 sec
	        Start  19: test stop gradient
	 19/264 Test  #19: test stop gradient ..................................................   Passed    0.12 sec
	        Start  20: test jvp
	 20/264 Test  #20: test jvp ............................................................   Passed    0.07 sec
	        Start  21: test vjp
	 21/264 Test  #21: test vjp ............................................................   Passed    0.08 sec
	        Start  22: test grad
	 22/264 Test  #22: test grad ...........................................................   Passed    0.12 sec
	        Start  23: test transform container reuse does not accumulate stale wrappers
	 23/264 Test  #23: test transform container reuse does not accumulate stale wrappers ...   Passed    0.09 sec
	        Start  24: test creation grads
	 24/264 Test  #24: test creation grads .................................................   Passed    0.11 sec
	        Start  25: test op vjps
	 25/264 Test  #25: test op vjps ........................................................   Passed    2.09 sec
	        Start  26: test gather and take grads
	 26/264 Test  #26: test gather and take grads ..........................................   Passed    1.25 sec
	        Start  27: test slice grads
	 27/264 Test  #27: test slice grads ....................................................   Passed    0.49 sec
	        Start  28: test min and max vjp
	 28/264 Test  #28: test min and max vjp ................................................   Passed    0.11 sec
	        Start  29: test reshape and transpose grads
	 29/264 Test  #29: test reshape and transpose grads ....................................   Passed    0.10 sec
	        Start  30: test copy grads
	 30/264 Test  #30: test copy grads .....................................................   Passed    0.11 sec
	        Start  31: test matmul vjp
	 31/264 Test  #31: test matmul vjp .....................................................   Passed    0.13 sec
	        Start  32: test concatenate grads
	 32/264 Test  #32: test concatenate grads ..............................................   Passed    0.11 sec
	        Start  33: test split grads
	 33/264 Test  #33: test split grads ....................................................   Passed    0.11 sec
	        Start  34: test comparison grads
	 34/264 Test  #34: test comparison grads ...............................................   Passed    0.14 sec
	        Start  35: test as_strided grads
	 35/264 Test  #35: test as_strided grads ...............................................   Passed    0.14 sec
	        Start  36: test jvp from vjp
	 36/264 Test  #36: test jvp from vjp ...................................................   Passed    2.20 sec
	        Start  37: test complex gradients
	 37/264 Test  #37: test complex gradients ..............................................   Passed    0.44 sec
	        Start  38: test scan grads
	 38/264 Test  #38: test scan grads .....................................................   Passed    0.19 sec
	        Start  39: test update state
	 39/264 Test  #39: test update state ...................................................   Passed    0.07 sec
	        Start  40: test grad types
	 40/264 Test  #40: test grad types .....................................................   Passed    0.06 sec
	        Start  41: test grad dynamic slices
	 41/264 Test  #41: test grad dynamic slices ............................................   Passed    0.12 sec
	        Start  42: test masked_scatter autograd
	 42/264 Test  #42: test masked_scatter autograd ........................................   Passed    0.11 sec
	        Start  43: test matmul
	 43/264 Test  #43: test matmul .........................................................   Passed    0.16 sec
	        Start  44: test simple compile
	 44/264 Test  #44: test simple compile .................................................   Passed    0.12 sec
	        Start  45: test compile with grad
	 45/264 Test  #45: test compile with grad ..............................................   Passed    0.11 sec
	        Start  46: test compile inputs with primitive
	 46/264 Test  #46: test compile inputs with primitive ..................................   Passed    0.80 sec
	        Start  47: test compile with created array
	 47/264 Test  #47: test compile with created array .....................................   Passed    0.07 sec
	        Start  48: test nested compile
	 48/264 Test  #48: test nested compile .................................................   Passed    0.09 sec
	        Start  49: test enable and disable compile
	 49/264 Test  #49: test enable and disable compile .....................................   Passed    0.05 sec
	        Start  50: test compile with non-finite constants
	 50/264 Test  #50: test compile with non-finite constants ..............................   Passed    0.09 sec
	        Start  51: test simplify scalars
	 51/264 Test  #51: test simplify scalars ...............................................   Passed    0.06 sec
	        Start  52: test simplify
	 52/264 Test  #52: test simplify .......................................................   Passed    0.07 sec
	        Start  53: test simplify noops
	 53/264 Test  #53: test simplify noops .................................................   Passed    0.07 sec
	        Start  54: test no simplify
	 54/264 Test  #54: test no simplify ....................................................   Passed    0.08 sec
	        Start  55: test simplify multi output
	 55/264 Test  #55: test simplify multi output ..........................................   Passed    0.10 sec
	        Start  56: test compile unary fused
	 56/264 Test  #56: test compile unary fused ............................................   Passed    0.11 sec
	        Start  57: test compile binary fused
	 57/264 Test  #57: test compile binary fused ...........................................   Passed    0.07 sec
	        Start  58: test compile gelu
	 58/264 Test  #58: test compile gelu ...................................................   Passed    0.12 sec
	        Start  59: test compile tape with outside parents
	 59/264 Test  #59: test compile tape with outside parents ..............................   Passed    0.12 sec
	        Start  60: test compile across streams
	 60/264 Test  #60: test compile across streams .........................................   Passed    0.07 sec
	        Start  61: test compile internal output
	 61/264 Test  #61: test compile internal output ........................................   Passed    0.10 sec
	        Start  62: test compile deep graph
	 62/264 Test  #62: test compile deep graph .............................................   Passed    0.13 sec
	        Start  63: test compile repeat input
	 63/264 Test  #63: test compile repeat input ...........................................   Passed    0.08 sec
	        Start  64: test compile compiled function
	 64/264 Test  #64: test compile compiled function ......................................   Passed    0.05 sec
	        Start  65: test transform compiled function
	 65/264 Test  #65: test transform compiled function ....................................   Passed    0.09 sec
	        Start  66: test fusion kernel reuse
	 66/264 Test  #66: test fusion kernel reuse ............................................   Passed    0.11 sec
	        Start  67: test fusion types
	 67/264 Test  #67: test fusion types ...................................................   Passed    0.08 sec
	        Start  68: test shapeless compile
	 68/264 Test  #68: test shapeless compile ..............................................   Passed    0.09 sec
	        Start  69: test compile strides
	 69/264 Test  #69: test compile strides ................................................   Passed    0.09 sec
	        Start  70: test compile change streams
	 70/264 Test  #70: test compile change streams .........................................   Passed    0.08 sec
	        Start  71: test compile lambda
	 71/264 Test  #71: test compile lambda .................................................   Passed    0.08 sec
	        Start  72: test compile with no-ops
	 72/264 Test  #72: test compile with no-ops ............................................   Passed    0.06 sec
	        Start  73: test compile random bits
	 73/264 Test  #73: test compile random bits ............................................   Passed    0.09 sec
	        Start  74: test compile throwing first trace does not poison cache
	 74/264 Test  #74: test compile throwing first trace does not poison cache .............   Passed    0.09 sec
	        Start  75: test arange
	 75/264 Test  #75: test arange .........................................................   Passed    0.10 sec
	        Start  76: test astype
	 76/264 Test  #76: test astype .........................................................   Passed    0.08 sec
	        Start  77: test full
	 77/264 Test  #77: test full ...........................................................   Passed    0.12 sec
	        Start  78: test simple custom vjp
	 78/264 Test  #78: test simple custom vjp ..............................................   Passed    0.09 sec
	        Start  79: test checkpointing
	 79/264 Test  #79: test checkpointing ..................................................   Passed    0.28 sec
	        Start  80: test device placement
	 80/264 Test  #80: test device placement ...............................................   Passed    0.07 sec
	        Start  81: test einsum path
	 81/264 Test  #81: test einsum path ....................................................   Passed    0.08 sec
	        Start  82: test einsum
	 82/264 Test  #82: test einsum .........................................................   Passed    0.15 sec
	        Start  83: test eval
	 83/264 Test  #83: test eval ...........................................................   Passed    0.08 sec
	        Start  84: test eval multiple
	 84/264 Test  #84: test eval multiple ..................................................   Passed    0.11 sec
	        Start  85: test eval with tracer when not tracing
	 85/264 Test  #85: test eval with tracer when not tracing ..............................   Passed    0.08 sec
	        Start  86: test eval graph retention when not tracing
	 86/264 Test  #86: test eval graph retention when not tracing ..........................   Passed    0.08 sec
	        Start  87: test export basic functions
	 87/264 Test  #87: test export basic functions .........................................   Passed    0.12 sec
	        Start  88: test export function with no inputs
	 88/264 Test  #88: test export function with no inputs .................................   Passed    0.10 sec
	        Start  89: test export multi output primitives
	 89/264 Test  #89: test export multi output primitives .................................   Passed    0.09 sec
	        Start  90: test export primitives with state
	 90/264 Test  #90: test export primitives with state ...................................   Passed    0.11 sec
	        Start  91: test export functions with kwargs
	 91/264 Test  #91: test export functions with kwargs ...................................   Passed    0.07 sec
	        Start  92: test export function with variable inputs
	 92/264 Test  #92: test export function with variable inputs ...........................   Passed    0.09 sec
	        Start  93: test export function on different stream
	 93/264 Test  #93: test export function on different stream ............................   Passed    0.08 sec
	        Start  94: test fft basics
	 94/264 Test  #94: test fft basics .....................................................   Passed    1.56 sec
	        Start  95: test real ffts
	 95/264 Test  #95: test real ffts ......................................................   Passed    0.28 sec
	        Start  96: test fftn
	 96/264 Test  #96: test fftn ...........................................................   Passed    0.82 sec
	        Start  97: test fft with provided shape
	 97/264 Test  #97: test fft with provided shape ........................................   Passed    0.07 sec
	        Start  98: test fft vmap
	 98/264 Test  #98: test fft vmap .......................................................   Passed    0.80 sec
	        Start  99: test fft grads
	 99/264 Test  #99: test fft grads ......................................................   Passed    1.14 sec
	        Start 100: test fftshift and ifftshift
	100/264 Test #100: test fftshift and ifftshift .........................................   Passed    0.10 sec
	        Start 101: test gpu arange
	101/264 Test #101: test gpu arange .....................................................   Passed    0.13 sec
	        Start 102: test gpu full
	102/264 Test #102: test gpu full .......................................................   Passed    0.09 sec
	        Start 103: test gpu astype
	103/264 Test #103: test gpu astype .....................................................   Passed    0.09 sec
	        Start 104: test gpu reshape
	104/264 Test #104: test gpu reshape ....................................................   Passed    0.08 sec
	        Start 105: test gpu reduce
	105/264 Test #105: test gpu reduce .....................................................   Passed    0.12 sec
	        Start 106: test gpu reduce with axes
	106/264 Test #106: test gpu reduce with axes ...........................................   Passed    0.10 sec
	        Start 107: test gpu binary ops
	107/264 Test #107: test gpu binary ops .................................................   Passed    0.12 sec
	        Start 108: test gpu unary ops
	108/264 Test #108: test gpu unary ops ..................................................   Passed    0.13 sec
	        Start 109: test gpu random
	109/264 Test #109: test gpu random .....................................................   Passed    0.06 sec
	        Start 110: test gpu matmul
	110/264 Test #110: test gpu matmul .....................................................   Passed    0.10 sec
	        Start 111: test gpu validation
	111/264 Test #111: test gpu validation .................................................   Passed    0.08 sec
	        Start 112: test gpu int32 shape overflow errors
	112/264 Test #112: test gpu int32 shape overflow errors ................................   Passed    0.06 sec
	        Start 113: test memory info
	113/264 Test #113: test memory info ....................................................   Passed    0.07 sec
	        Start 114: test scatter_prod with NaN does not hang
	114/264 Test #114: test scatter_prod with NaN does not hang ............................   Passed    0.07 sec
	        Start 115: test gpu depthwise conv2d non-mod-8 spatial
	115/264 Test #115: test gpu depthwise conv2d non-mod-8 spatial .........................   Passed    0.22 sec
	        Start 116: test layer norm vjp bias grad race
	116/264 Test #116: test layer norm vjp bias grad race ..................................   Passed    4.40 sec
	        Start 117: [mlx.core.linalg.norm] no ord
	117/264 Test #117: [mlx.core.linalg.norm] no ord .......................................   Passed    0.34 sec
	        Start 118: [mlx.core.linalg.norm] double ord
	118/264 Test #118: [mlx.core.linalg.norm] double ord ...................................   Passed    0.65 sec
	        Start 119: [mlx.core.linalg.norm] string ord
	119/264 Test #119: [mlx.core.linalg.norm] string ord ...................................   Passed    1.56 sec
	        Start 120: test QR factorization
	120/264 Test #120: test QR factorization ...............................................   Passed    0.13 sec
	        Start 121: test SVD factorization
	121/264 Test #121: test SVD factorization ..............................................   Passed    0.70 sec
	        Start 122: test matrix inversion
	122/264 Test #122: test matrix inversion ...............................................   Passed    0.36 sec
	        Start 123: test matrix cholesky
	123/264 Test #123: test matrix cholesky ................................................   Passed    0.30 sec
	        Start 124: test matrix pseudo-inverse
	124/264 Test #124: test matrix pseudo-inverse ..........................................   Passed    0.58 sec
	        Start 125: test cross product
	125/264 Test #125: test cross product ..................................................   Passed    0.13 sec
	        Start 126: test matrix eigh
	126/264 Test #126: test matrix eigh ....................................................   Passed    0.15 sec
	        Start 127: test lu
	127/264 Test #127: test lu .............................................................   Passed    0.65 sec
	        Start 128: test solve
	128/264 Test #128: test solve ..........................................................   Passed    0.14 sec
	        Start 129: test solve_triangluar
	129/264 Test #129: test solve_triangluar ...............................................   Passed    0.08 sec
	        Start 130: test det
	130/264 Test #130: test det ............................................................   Passed    0.07 sec
	        Start 131: test slogdet
	131/264 Test #131: test slogdet ........................................................   Passed    0.08 sec
	        Start 132: test save_safetensors
	132/264 Test #132: test save_safetensors ...............................................   Passed    0.07 sec
	        Start 133: test safetensors file boundary validation
	133/264 Test #133: test safetensors file boundary validation ...........................   Passed    0.05 sec
	        Start 134: test gguf
	134/264 Test #134: test gguf ...........................................................   Passed    0.13 sec
	        Start 135: test gguf metadata
	135/264 Test #135: test gguf metadata ..................................................   Passed    0.10 sec
	        Start 136: test single array serialization
	136/264 Test #136: test single array serialization .....................................   Passed    0.77 sec
	        Start 137: test copy
	137/264 Test #137: test copy ...........................................................   Passed    0.09 sec
	        Start 138: test reshape
	138/264 Test #138: test reshape ........................................................   Passed    0.08 sec
	        Start 139: test flatten
	139/264 Test #139: test flatten ........................................................   Passed    0.06 sec
	        Start 140: test unflatten
	140/264 Test #140: test unflatten ......................................................   Passed    0.06 sec
	        Start 141: test squeeze and expand
	141/264 Test #141: test squeeze and expand .............................................   Passed    0.06 sec
	        Start 142: test slice
	142/264 Test #142: test slice ..........................................................   Passed    0.06 sec
	        Start 143: test slice update
	143/264 Test #143: test slice update ...................................................   Passed    0.05 sec
	        Start 144: test slice update add
	144/264 Test #144: test slice update add ...............................................   Passed    0.12 sec
	        Start 145: test dynamic slice
	145/264 Test #145: test dynamic slice ..................................................   Passed    0.08 sec
	        Start 146: test dynamic slice update
	146/264 Test #146: test dynamic slice update ...........................................   Passed    0.09 sec
	        Start 147: test split
	147/264 Test #147: test split ..........................................................   Passed    0.10 sec
	        Start 148: test flip
	148/264 Test #148: test flip ...........................................................   Passed    0.07 sec
	        Start 149: test unstack
	149/264 Test #149: test unstack ........................................................   Passed    0.11 sec
	        Start 150: test swap and move axes
	150/264 Test #150: test swap and move axes .............................................   Passed    0.07 sec
	        Start 151: test transpose
	151/264 Test #151: test transpose ......................................................   Passed    0.11 sec
	        Start 152: test comparison ops
	152/264 Test #152: test comparison ops .................................................   Passed    0.16 sec
	        Start 153: test is nan
	153/264 Test #153: test is nan .........................................................   Passed    0.10 sec
	        Start 154: test is inf
	154/264 Test #154: test is inf .........................................................   Passed    0.13 sec
	        Start 155: test all close
	155/264 Test #155: test all close ......................................................   Passed    0.13 sec
	        Start 156: test is close
	156/264 Test #156: test is close .......................................................   Passed    0.15 sec
	        Start 157: test reduction ops
	157/264 Test #157: test reduction ops ..................................................   Passed    1.06 sec
	        Start 158: test irregular binary ops
	158/264 Test #158: test irregular binary ops ...........................................   Passed    0.09 sec
	        Start 159: test arithmetic unary ops
	159/264 Test #159: test arithmetic unary ops ...........................................   Passed    0.54 sec
	        Start 160: test error functions
	160/264 Test #160: test error functions ................................................   Passed    0.09 sec
	        Start 161: test arithmetic binary ops
	161/264 Test #161: test arithmetic binary ops ..........................................   Passed    0.30 sec
	        Start 162: test broadcast
	162/264 Test #162: test broadcast ......................................................   Passed    0.06 sec
	        Start 163: test gather
	163/264 Test #163: test gather .........................................................   Passed    0.60 sec
	        Start 164: test take
	164/264 Test #164: test take ...........................................................   Passed    2.46 sec
	        Start 165: test gather contiguity
	165/264 Test #165: test gather contiguity ..............................................   Passed    0.07 sec
	        Start 166: test take along axis
	166/264 Test #166: test take along axis ................................................   Passed    1.82 sec
	        Start 167: test put along axis
	167/264 Test #167: test put along axis .................................................   Passed    0.11 sec
	        Start 168: test scatter
	168/264 Test #168: test scatter ........................................................   Passed    0.30 sec
	        Start 169: test masked_scatter
	169/264 Test #169: test masked_scatter .................................................   Passed    0.09 sec
	        Start 170: test is positive infinity
	170/264 Test #170: test is positive infinity ...........................................   Passed    0.11 sec
	        Start 171: test is negative infinity
	171/264 Test #171: test is negative infinity ...........................................   Passed    0.10 sec
	        Start 172: test scatter types
	172/264 Test #172: test scatter types ..................................................   Passed    0.15 sec
	        Start 173: test complex ops
	173/264 Test #173: test complex ops ....................................................   Passed    1.21 sec
	        Start 174: test as_strided op
	174/264 Test #174: test as_strided op ..................................................   Passed    0.09 sec
	        Start 175: test scan op
	175/264 Test #175: test scan op ........................................................   Passed    0.08 sec
	        Start 176: test pad
	176/264 Test #176: test pad ............................................................   Passed    0.08 sec
	        Start 177: test power
	177/264 Test #177: test power ..........................................................   Passed    0.56 sec
	        Start 178: test where
	178/264 Test #178: test where ..........................................................   Passed    0.12 sec
	        Start 179: test stack
	179/264 Test #179: test stack ..........................................................   Passed    0.05 sec
	        Start 180: test full_like
	180/264 Test #180: test full_like ......................................................   Passed    0.10 sec
	        Start 181: test eye
	181/264 Test #181: test eye ............................................................   Passed    0.10 sec
	        Start 182: test tri
	182/264 Test #182: test tri ............................................................   Passed    0.09 sec
	        Start 183: test tril
	183/264 Test #183: test tril ...........................................................   Passed    0.07 sec
	        Start 184: test triu
	184/264 Test #184: test triu ...........................................................   Passed    0.07 sec
	        Start 185: test identity
	185/264 Test #185: test identity .......................................................   Passed    0.09 sec
	        Start 186: test eye with positive k offset
	186/264 Test #186: test eye with positive k offset .....................................   Passed    0.06 sec
	        Start 187: test eye with negative k offset
	187/264 Test #187: test eye with negative k offset .....................................   Passed    0.07 sec
	        Start 188: test basic clipping
	188/264 Test #188: test basic clipping .................................................   Passed    0.27 sec
	        Start 189: test clipping with only min
	189/264 Test #189: test clipping with only min .........................................   Passed    0.07 sec
	        Start 190: test clipping with only max
	190/264 Test #190: test clipping with only max .........................................   Passed    0.26 sec
	        Start 191: test linspace
	191/264 Test #191: test linspace .......................................................   Passed    0.11 sec
	        Start 192: test quantize dequantize
	192/264 Test #192: test quantize dequantize ............................................   Passed    0.09 sec
	        Start 193: test repeat
	193/264 Test #193: test repeat .........................................................   Passed    0.10 sec
	        Start 194: tile
	194/264 Test #194: tile ................................................................   Passed    0.07 sec
	        Start 195: tensordot
	195/264 Test #195: tensordot ...........................................................   Passed    0.12 sec
	        Start 196: outer
	196/264 Test #196: outer ...............................................................   Passed    0.07 sec
	        Start 197: inner
	197/264 Test #197: inner ...............................................................   Passed    0.12 sec
	        Start 198: test divmod
	198/264 Test #198: test divmod .........................................................   Passed    0.08 sec
	        Start 199: test diagonal
	199/264 Test #199: test diagonal .......................................................   Passed    0.12 sec
	        Start 200: test diag
	200/264 Test #200: test diag ...........................................................   Passed    0.11 sec
	        Start 201: test issubdtype
	201/264 Test #201: test issubdtype .....................................................   Passed    0.07 sec
	        Start 202: test atleast_1d
	202/264 Test #202: test atleast_1d .....................................................   Passed    0.06 sec
	        Start 203: test atleast_1d vector
	203/264 Test #203: test atleast_1d vector ..............................................   Passed    0.06 sec
	        Start 204: test atleast_2d
	204/264 Test #204: test atleast_2d .....................................................   Passed    0.07 sec
	        Start 205: test atleast_2d vector
	205/264 Test #205: test atleast_2d vector ..............................................   Passed    0.06 sec
	        Start 206: test atleast_3d
	206/264 Test #206: test atleast_3d .....................................................   Passed    0.03 sec
	        Start 207: test atleast_3d vector
	207/264 Test #207: test atleast_3d vector ..............................................   Passed    0.05 sec
	        Start 208: test topk
	208/264 Test #208: test topk ...........................................................   Passed    0.05 sec
	        Start 209: test meshgrid
	209/264 Test #209: test meshgrid .......................................................   Passed    0.09 sec
	        Start 210: test conv1d
	210/264 Test #210: test conv1d .........................................................   Passed    0.12 sec
	        Start 211: test conv2d
	211/264 Test #211: test conv2d .........................................................   Passed    0.34 sec
	        Start 212: test trace
	212/264 Test #212: test trace ..........................................................   Passed    0.12 sec
	        Start 213: test view
	213/264 Test #213: test view ...........................................................   Passed    0.08 sec
	        Start 214: test roll
	214/264 Test #214: test roll ...........................................................   Passed    0.09 sec
	        Start 215: test contiguous
	215/264 Test #215: test contiguous .....................................................   Passed    0.07 sec
	        Start 216: test bitwise shift operations
	216/264 Test #216: test bitwise shift operations .......................................   Passed    0.14 sec
	        Start 217: test conv_transpose1d with output_padding
	217/264 Test #217: test conv_transpose1d with output_padding ...........................   Passed    0.06 sec
	        Start 218: test conv_transpose2d with output_padding
	218/264 Test #218: test conv_transpose2d with output_padding ...........................   Passed    0.07 sec
	        Start 219: test conv_transpose3d with output_padding
	219/264 Test #219: test conv_transpose3d with output_padding ...........................   Passed    0.07 sec
	        Start 220: test fp8 conversion
	220/264 Test #220: test fp8 conversion .................................................   Passed    0.06 sec
	        Start 221: test max min with nan
	221/264 Test #221: test max min with nan ...............................................   Passed    0.11 sec
	        Start 222: roll and tile shape overflow
	222/264 Test #222: roll and tile shape overflow ........................................   Passed    0.08 sec
	        Start 223: test random key
	223/264 Test #223: test random key .....................................................   Passed    0.11 sec
	        Start 224: test global rng
	224/264 Test #224: test global rng .....................................................   Passed    0.08 sec
	        Start 225: test random split
	225/264 Test #225: test random split ...................................................   Passed    0.08 sec
	        Start 226: test random bits
	226/264 Test #226: test random bits ....................................................   Passed    0.72 sec
	        Start 227: test random uniform
	227/264 Test #227: test random uniform .................................................   Passed    0.76 sec
	        Start 228: test random normal
	228/264 Test #228: test random normal ..................................................   Passed    0.59 sec
	        Start 229: test random multivariate_normal
	229/264 Test #229: test random multivariate_normal .....................................   Passed    0.06 sec
	        Start 230: test random randint
	230/264 Test #230: test random randint .................................................   Passed    0.56 sec
	        Start 231: test random bernoulli
	231/264 Test #231: test random bernoulli ...............................................   Passed    0.09 sec
	        Start 232: Test truncated normal
	232/264 Test #232: Test truncated normal ...............................................   Passed    0.45 sec
	        Start 233: test categorical
	233/264 Test #233: test categorical ....................................................   Passed    0.73 sec
	        Start 234: test laplace
	234/264 Test #234: test laplace ........................................................   Passed    0.86 sec
	        Start 235: test stream management
	235/264 Test #235: test stream management ..............................................   Passed    0.05 sec
	        Start 236: test default stream in threads
	236/264 Test #236: test default stream in threads ......................................   Passed    0.03 sec
	        Start 237: test access stream in other thread
	237/264 Test #237: test access stream in other thread ..................................   Passed    0.06 sec
	        Start 238: test new stream in threads
	238/264 Test #238: test new stream in threads ..........................................   Passed    0.07 sec
	        Start 239: test thread unsafe stream
	239/264 Test #239: test thread unsafe stream ...........................................   Passed    0.08 sec
	        Start 240: test thread local stream
	240/264 Test #240: test thread local stream ............................................   Passed    0.06 sec
	        Start 241: test get streams
	241/264 Test #241: test get streams ....................................................   Passed    0.03 sec
	        Start 242: test asynchronous launch
	242/264 Test #242: test asynchronous launch ............................................   Passed    0.06 sec
	        Start 243: test stream placement
	243/264 Test #243: test stream placement ...............................................   Passed    0.07 sec
	        Start 244: test scheduler races
	244/264 Test #244: test scheduler races ................................................   Passed    0.83 sec
	        Start 245: test type promotion
	245/264 Test #245: test type promotion .................................................   Passed    0.06 sec
	        Start 246: test normalize axis
	246/264 Test #246: test normalize axis .................................................   Passed    0.06 sec
	        Start 247: test finfo
	247/264 Test #247: test finfo ..........................................................   Passed    0.06 sec
	        Start 248: test iinfo
	248/264 Test #248: test iinfo ..........................................................   Passed    0.06 sec
	        Start 249: test simple vmap
	249/264 Test #249: test simple vmap ....................................................   Passed    0.14 sec
	        Start 250: test vmap with eval
	250/264 Test #250: test vmap with eval .................................................   Passed    0.09 sec
	        Start 251: test vmap comparison ops
	251/264 Test #251: test vmap comparison ops ............................................   Passed    0.06 sec
	        Start 252: test vmap creation ops
	252/264 Test #252: test vmap creation ops ..............................................   Passed    0.07 sec
	        Start 253: test vmap slice
	253/264 Test #253: test vmap slice .....................................................   Passed    0.08 sec
	        Start 254: test vmap concatenate
	254/264 Test #254: test vmap concatenate ...............................................   Passed    0.09 sec
	        Start 255: test vmap gather
	255/264 Test #255: test vmap gather ....................................................   Passed    0.06 sec
	        Start 256: test vmap take_along_axis with unmapped input and mapped index
	256/264 Test #256: test vmap take_along_axis with unmapped input and mapped index ......   Passed    0.07 sec
	        Start 257: test vmap scatter
	257/264 Test #257: test vmap scatter ...................................................   Passed    0.09 sec
	        Start 258: test vmap SVD
	258/264 Test #258: test vmap SVD .......................................................   Passed    0.04 sec
	        Start 259: test vmap dynamic slices
	259/264 Test #259: test vmap dynamic slices ............................................   Passed    0.11 sec
	        Start 260: test vmap floor_divide integer
	260/264 Test #260: test vmap floor_divide integer ......................................   Passed    0.12 sec
	        Start 261: test vulkan complex scalar view multiply regression
	261/264 Test #261: test vulkan complex scalar view multiply regression .................   Passed    0.09 sec
	        Start 262: test vulkan complex abs general layout regression
	262/264 Test #262: test vulkan complex abs general layout regression ...................   Passed    0.06 sec
	        Start 263: tests
	263/264 Test #263: tests ...............................................................   Passed   32.77 sec
	        Start 264: teardown
	264/264 Test #264: teardown ............................................................   Passed    1.06 sec
	
	100% tests passed, 0 tests failed out of 264
	
	Total Test time (real) =  91.21 sec



---
2026.7.27

All fixed for basci primtives which may contain certain roll backs to CPUs.

Next: need to add C++ test cases for FAST primitives.  

	Running tests...
	Test project /home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl726/mlx-opencl/build
	Connected to MAKE jobserver
	        Start   1: test simple allocations
	  1/251 Test   #1: test simple allocations .............................................   Passed    0.04 sec
	        Start   2: test large allocations
	  2/251 Test   #2: test large allocations ..............................................   Passed    0.16 sec
	        Start   3: test arg reduce small
	  3/251 Test   #3: test arg reduce small ...............................................   Passed    0.06 sec
	        Start   4: test arg reduce against cpu
	  4/251 Test   #4: test arg reduce against cpu .........................................   Passed    0.05 sec
	        Start   5: test arg reduce bool
	  5/251 Test   #5: test arg reduce bool ................................................   Passed    0.03 sec
	        Start   6: test arg reduce edge cases
	  6/251 Test   #6: test arg reduce edge cases ..........................................   Passed    0.07 sec
	        Start   7: test arg reduce irregular strides
	  7/251 Test   #7: test arg reduce irregular strides ...................................   Passed    0.06 sec
	        Start   8: test array basics
	  8/251 Test   #8: test array basics ...................................................   Passed    0.05 sec
	        Start   9: test array types
	  9/251 Test   #9: test array types ....................................................   Passed    0.05 sec
	        Start  10: test array metadata
	 10/251 Test  #10: test array metadata .................................................   Passed    0.09 sec
	        Start  11: test array iteration
	 11/251 Test  #11: test array iteration ................................................   Passed    0.07 sec
	        Start  12: test array shared buffer
	 12/251 Test  #12: test array shared buffer ............................................   Passed    0.07 sec
	        Start  13: test make empty array
	 13/251 Test  #13: test make empty array ...............................................   Passed    0.04 sec
	        Start  14: test make array from user buffer
	 14/251 Test  #14: test make array from user buffer ....................................   Passed    0.08 sec
	        Start  15: test negative indexing for shape/strides
	 15/251 Test  #15: test negative indexing for shape/strides ............................   Passed    0.05 sec
	        Start  16: test stop gradient
	 16/251 Test  #16: test stop gradient ..................................................   Passed    0.10 sec
	        Start  17: test jvp
	 17/251 Test  #17: test jvp ............................................................   Passed    0.06 sec
	        Start  18: test vjp
	 18/251 Test  #18: test vjp ............................................................   Passed    0.06 sec
	        Start  19: test grad
	 19/251 Test  #19: test grad ...........................................................   Passed    0.07 sec
	        Start  20: test transform container reuse does not accumulate stale wrappers
	 20/251 Test  #20: test transform container reuse does not accumulate stale wrappers ...   Passed    0.06 sec
	        Start  21: test creation grads
	 21/251 Test  #21: test creation grads .................................................   Passed    0.07 sec
	        Start  22: test op vjps
	 22/251 Test  #22: test op vjps ........................................................   Passed    1.89 sec
	        Start  23: test gather and take grads
	 23/251 Test  #23: test gather and take grads ..........................................   Passed    1.20 sec
	        Start  24: test slice grads
	 24/251 Test  #24: test slice grads ....................................................   Passed    0.43 sec
	        Start  25: test min and max vjp
	 25/251 Test  #25: test min and max vjp ................................................   Passed    0.12 sec
	        Start  26: test reshape and transpose grads
	 26/251 Test  #26: test reshape and transpose grads ....................................   Passed    0.11 sec
	        Start  27: test copy grads
	 27/251 Test  #27: test copy grads .....................................................   Passed    0.09 sec
	        Start  28: test matmul vjp
	 28/251 Test  #28: test matmul vjp .....................................................   Passed    0.10 sec
	        Start  29: test concatenate grads
	 29/251 Test  #29: test concatenate grads ..............................................   Passed    0.07 sec
	        Start  30: test split grads
	 30/251 Test  #30: test split grads ....................................................   Passed    0.07 sec
	        Start  31: test comparison grads
	 31/251 Test  #31: test comparison grads ...............................................   Passed    0.10 sec
	        Start  32: test as_strided grads
	 32/251 Test  #32: test as_strided grads ...............................................   Passed    0.12 sec
	        Start  33: test jvp from vjp
	 33/251 Test  #33: test jvp from vjp ...................................................   Passed    2.14 sec
	        Start  34: test complex gradients
	 34/251 Test  #34: test complex gradients ..............................................   Passed    0.42 sec
	        Start  35: test scan grads
	 35/251 Test  #35: test scan grads .....................................................   Passed    0.15 sec
	        Start  36: test update state
	 36/251 Test  #36: test update state ...................................................   Passed    0.08 sec
	        Start  37: test grad types
	 37/251 Test  #37: test grad types .....................................................   Passed    0.07 sec
	        Start  38: test grad dynamic slices
	 38/251 Test  #38: test grad dynamic slices ............................................   Passed    0.13 sec
	        Start  39: test masked_scatter autograd
	 39/251 Test  #39: test masked_scatter autograd ........................................   Passed    0.09 sec
	        Start  40: test matmul
	 40/251 Test  #40: test matmul .........................................................   Passed    0.16 sec
	        Start  41: test simple compile
	 41/251 Test  #41: test simple compile .................................................   Passed    0.13 sec
	        Start  42: test compile with grad
	 42/251 Test  #42: test compile with grad ..............................................   Passed    0.10 sec
	        Start  43: test compile inputs with primitive
	 43/251 Test  #43: test compile inputs with primitive ..................................   Passed    0.78 sec
	        Start  44: test compile with created array
	 44/251 Test  #44: test compile with created array .....................................   Passed    0.06 sec
	        Start  45: test nested compile
	 45/251 Test  #45: test nested compile .................................................   Passed    0.07 sec
	        Start  46: test enable and disable compile
	 46/251 Test  #46: test enable and disable compile .....................................   Passed    0.06 sec
	        Start  47: test simplify scalars
	 47/251 Test  #47: test simplify scalars ...............................................   Passed    0.06 sec
	        Start  48: test simplify
	 48/251 Test  #48: test simplify .......................................................   Passed    0.05 sec
	        Start  49: test simplify noops
	 49/251 Test  #49: test simplify noops .................................................   Passed    0.03 sec
	        Start  50: test no simplify
	 50/251 Test  #50: test no simplify ....................................................   Passed    0.05 sec
	        Start  51: test simplify multi output
	 51/251 Test  #51: test simplify multi output ..........................................   Passed    0.05 sec
	        Start  52: test compile unary fused
	 52/251 Test  #52: test compile unary fused ............................................   Passed    0.11 sec
	        Start  53: test compile binary fused
	 53/251 Test  #53: test compile binary fused ...........................................   Passed    0.06 sec
	        Start  54: test compile gelu
	 54/251 Test  #54: test compile gelu ...................................................   Passed    0.11 sec
	        Start  55: test compile tape with outside parents
	 55/251 Test  #55: test compile tape with outside parents ..............................   Passed    0.14 sec
	        Start  56: test compile across streams
	 56/251 Test  #56: test compile across streams .........................................   Passed    0.05 sec
	        Start  57: test compile internal output
	 57/251 Test  #57: test compile internal output ........................................   Passed    0.07 sec
	        Start  58: test compile deep graph
	 58/251 Test  #58: test compile deep graph .............................................   Passed    0.11 sec
	        Start  59: test compile repeat input
	 59/251 Test  #59: test compile repeat input ...........................................   Passed    0.09 sec
	        Start  60: test compile compiled function
	 60/251 Test  #60: test compile compiled function ......................................   Passed    0.06 sec
	        Start  61: test transform compiled function
	 61/251 Test  #61: test transform compiled function ....................................   Passed    0.03 sec
	        Start  62: test fusion kernel reuse
	 62/251 Test  #62: test fusion kernel reuse ............................................   Passed    0.08 sec
	        Start  63: test fusion types
	 63/251 Test  #63: test fusion types ...................................................   Passed    0.06 sec
	        Start  64: test shapeless compile
	 64/251 Test  #64: test shapeless compile ..............................................   Passed    0.08 sec
	        Start  65: test compile strides
	 65/251 Test  #65: test compile strides ................................................   Passed    0.08 sec
	        Start  66: test compile change streams
	 66/251 Test  #66: test compile change streams .........................................   Passed    0.06 sec
	        Start  67: test compile lambda
	 67/251 Test  #67: test compile lambda .................................................   Passed    0.05 sec
	        Start  68: test compile with no-ops
	 68/251 Test  #68: test compile with no-ops ............................................   Passed    0.06 sec
	        Start  69: test compile random bits
	 69/251 Test  #69: test compile random bits ............................................   Passed    0.07 sec
	        Start  70: test arange
	 70/251 Test  #70: test arange .........................................................   Passed    0.12 sec
	        Start  71: test astype
	 71/251 Test  #71: test astype .........................................................   Passed    0.07 sec
	        Start  72: test full
	 72/251 Test  #72: test full ...........................................................   Passed    0.11 sec
	        Start  73: test simple custom vjp
	 73/251 Test  #73: test simple custom vjp ..............................................   Passed    0.10 sec
	        Start  74: test checkpointing
	 74/251 Test  #74: test checkpointing ..................................................   Passed    0.24 sec
	        Start  75: test device placement
	 75/251 Test  #75: test device placement ...............................................   Passed    0.06 sec
	        Start  76: test einsum path
	 76/251 Test  #76: test einsum path ....................................................   Passed    0.05 sec
	        Start  77: test einsum
	 77/251 Test  #77: test einsum .........................................................   Passed    0.13 sec
	        Start  78: test eval
	 78/251 Test  #78: test eval ...........................................................   Passed    0.08 sec
	        Start  79: test eval multiple
	 79/251 Test  #79: test eval multiple ..................................................   Passed    0.11 sec
	        Start  80: test eval with tracer when not tracing
	 80/251 Test  #80: test eval with tracer when not tracing ..............................   Passed    0.07 sec
	        Start  81: test eval graph retention when not tracing
	 81/251 Test  #81: test eval graph retention when not tracing ..........................   Passed    0.06 sec
	        Start  82: test export basic functions
	 82/251 Test  #82: test export basic functions .........................................   Passed    0.11 sec
	        Start  83: test export function with no inputs
	 83/251 Test  #83: test export function with no inputs .................................   Passed    0.10 sec
	        Start  84: test export multi output primitives
	 84/251 Test  #84: test export multi output primitives .................................   Passed    0.10 sec
	        Start  85: test export primitives with state
	 85/251 Test  #85: test export primitives with state ...................................   Passed    0.06 sec
	        Start  86: test export functions with kwargs
	 86/251 Test  #86: test export functions with kwargs ...................................   Passed    0.05 sec
	        Start  87: test export function with variable inputs
	 87/251 Test  #87: test export function with variable inputs ...........................   Passed    0.07 sec
	        Start  88: test export function on different stream
	 88/251 Test  #88: test export function on different stream ............................   Passed    0.04 sec
	        Start  89: test fft basics
	 89/251 Test  #89: test fft basics .....................................................   Passed    1.54 sec
	        Start  90: test real ffts
	 90/251 Test  #90: test real ffts ......................................................   Passed    0.26 sec
	        Start  91: test fftn
	 91/251 Test  #91: test fftn ...........................................................   Passed    0.75 sec
	        Start  92: test fft with provided shape
	 92/251 Test  #92: test fft with provided shape ........................................   Passed    0.05 sec
	        Start  93: test fft vmap
	 93/251 Test  #93: test fft vmap .......................................................   Passed    0.72 sec
	        Start  94: test fft grads
	 94/251 Test  #94: test fft grads ......................................................   Passed    1.08 sec
	        Start  95: test fftshift and ifftshift
	 95/251 Test  #95: test fftshift and ifftshift .........................................   Passed    0.11 sec
	        Start  96: test gpu arange
	 96/251 Test  #96: test gpu arange .....................................................   Passed    0.07 sec
	        Start  97: test gpu full
	 97/251 Test  #97: test gpu full .......................................................   Passed    0.07 sec
	        Start  98: test gpu astype
	 98/251 Test  #98: test gpu astype .....................................................   Passed    0.08 sec
	        Start  99: test gpu reshape
	 99/251 Test  #99: test gpu reshape ....................................................   Passed    0.07 sec
	        Start 100: test gpu reduce
	100/251 Test #100: test gpu reduce .....................................................   Passed    0.11 sec
	        Start 101: test gpu reduce with axes
	101/251 Test #101: test gpu reduce with axes ...........................................   Passed    0.07 sec
	        Start 102: test gpu binary ops
	102/251 Test #102: test gpu binary ops .................................................   Passed    0.10 sec
	        Start 103: test gpu unary ops
	103/251 Test #103: test gpu unary ops ..................................................   Passed    0.10 sec
	        Start 104: test gpu random
	104/251 Test #104: test gpu random .....................................................   Passed    0.06 sec
	        Start 105: test gpu matmul
	105/251 Test #105: test gpu matmul .....................................................   Passed    0.12 sec
	        Start 106: test gpu validation
	106/251 Test #106: test gpu validation .................................................   Passed    0.09 sec
	        Start 107: test memory info
	107/251 Test #107: test memory info ....................................................   Passed    0.08 sec
	        Start 108: test scatter_prod with NaN does not hang
	108/251 Test #108: test scatter_prod with NaN does not hang ............................   Passed    0.07 sec
	        Start 109: [mlx.core.linalg.norm] no ord
	109/251 Test #109: [mlx.core.linalg.norm] no ord .......................................   Passed    0.33 sec
	        Start 110: [mlx.core.linalg.norm] double ord
	110/251 Test #110: [mlx.core.linalg.norm] double ord ...................................   Passed    0.64 sec
	        Start 111: [mlx.core.linalg.norm] string ord
	111/251 Test #111: [mlx.core.linalg.norm] string ord ...................................   Passed    1.46 sec
	        Start 112: test QR factorization
	112/251 Test #112: test QR factorization ...............................................   Passed    0.13 sec
	        Start 113: test SVD factorization
	113/251 Test #113: test SVD factorization ..............................................   Passed    0.62 sec
	        Start 114: test matrix inversion
	114/251 Test #114: test matrix inversion ...............................................   Passed    0.29 sec
	        Start 115: test matrix cholesky
	115/251 Test #115: test matrix cholesky ................................................   Passed    0.27 sec
	        Start 116: test matrix pseudo-inverse
	116/251 Test #116: test matrix pseudo-inverse ..........................................   Passed    0.49 sec
	        Start 117: test cross product
	117/251 Test #117: test cross product ..................................................   Passed    0.13 sec
	        Start 118: test matrix eigh
	118/251 Test #118: test matrix eigh ....................................................   Passed    0.15 sec
	        Start 119: test lu
	119/251 Test #119: test lu .............................................................   Passed    0.71 sec
	        Start 120: test solve
	120/251 Test #120: test solve ..........................................................   Passed    0.16 sec
	        Start 121: test solve_triangluar
	121/251 Test #121: test solve_triangluar ...............................................   Passed    0.11 sec
	        Start 122: test det
	122/251 Test #122: test det ............................................................   Passed    0.08 sec
	        Start 123: test slogdet
	123/251 Test #123: test slogdet ........................................................   Passed    0.09 sec
	        Start 124: test save_safetensors
	124/251 Test #124: test save_safetensors ...............................................   Passed    0.09 sec
	        Start 125: test safetensors file boundary validation
	125/251 Test #125: test safetensors file boundary validation ...........................   Passed    0.07 sec
	        Start 126: test gguf
	126/251 Test #126: test gguf ...........................................................   Passed    0.11 sec
	        Start 127: test gguf metadata
	127/251 Test #127: test gguf metadata ..................................................   Passed    0.09 sec
	        Start 128: test single array serialization
	128/251 Test #128: test single array serialization .....................................   Passed    0.59 sec
	        Start 129: test copy
	129/251 Test #129: test copy ...........................................................   Passed    0.07 sec
	        Start 130: test reshape
	130/251 Test #130: test reshape ........................................................   Passed    0.06 sec
	        Start 131: test flatten
	131/251 Test #131: test flatten ........................................................   Passed    0.06 sec
	        Start 132: test unflatten
	132/251 Test #132: test unflatten ......................................................   Passed    0.05 sec
	        Start 133: test squeeze and expand
	133/251 Test #133: test squeeze and expand .............................................   Passed    0.06 sec
	        Start 134: test slice
	134/251 Test #134: test slice ..........................................................   Passed    0.11 sec
	        Start 135: test slice update
	135/251 Test #135: test slice update ...................................................   Passed    0.10 sec
	        Start 136: test slice update add
	136/251 Test #136: test slice update add ...............................................   Passed    0.10 sec
	        Start 137: test dynamic slice
	137/251 Test #137: test dynamic slice ..................................................   Passed    0.09 sec
	        Start 138: test dynamic slice update
	138/251 Test #138: test dynamic slice update ...........................................   Passed    0.10 sec
	        Start 139: test split
	139/251 Test #139: test split ..........................................................   Passed    0.09 sec
	        Start 140: test swap and move axes
	140/251 Test #140: test swap and move axes .............................................   Passed    0.06 sec
	        Start 141: test transpose
	141/251 Test #141: test transpose ......................................................   Passed    0.09 sec
	        Start 142: test comparison ops
	142/251 Test #142: test comparison ops .................................................   Passed    0.13 sec
	        Start 143: test is nan
	143/251 Test #143: test is nan .........................................................   Passed    0.10 sec
	        Start 144: test is inf
	144/251 Test #144: test is inf .........................................................   Passed    0.11 sec
	        Start 145: test all close
	145/251 Test #145: test all close ......................................................   Passed    0.13 sec
	        Start 146: test is close
	146/251 Test #146: test is close .......................................................   Passed    0.14 sec
	        Start 147: test reduction ops
	147/251 Test #147: test reduction ops ..................................................   Passed    1.18 sec
	        Start 148: test irregular binary ops
	148/251 Test #148: test irregular binary ops ...........................................   Passed    0.10 sec
	        Start 149: test arithmetic unary ops
	149/251 Test #149: test arithmetic unary ops ...........................................   Passed    0.70 sec
	        Start 150: test error functions
	150/251 Test #150: test error functions ................................................   Passed    0.11 sec
	        Start 151: test arithmetic binary ops
	151/251 Test #151: test arithmetic binary ops ..........................................   Passed    0.28 sec
	        Start 152: test broadcast
	152/251 Test #152: test broadcast ......................................................   Passed    0.09 sec
	        Start 153: test gather
	153/251 Test #153: test gather .........................................................   Passed    0.59 sec
	        Start 154: test take
	154/251 Test #154: test take ...........................................................   Passed    2.53 sec
	        Start 155: test take along axis
	155/251 Test #155: test take along axis ................................................   Passed    1.97 sec
	        Start 156: test put along axis
	156/251 Test #156: test put along axis .................................................   Passed    0.09 sec
	        Start 157: test scatter
	157/251 Test #157: test scatter ........................................................   Passed    0.31 sec
	        Start 158: test masked_scatter
	158/251 Test #158: test masked_scatter .................................................   Passed    0.09 sec
	        Start 159: test is positive infinity
	159/251 Test #159: test is positive infinity ...........................................   Passed    0.11 sec
	        Start 160: test is negative infinity
	160/251 Test #160: test is negative infinity ...........................................   Passed    0.11 sec
	        Start 161: test scatter types
	161/251 Test #161: test scatter types ..................................................   Passed    0.15 sec
	        Start 162: test complex ops
	162/251 Test #162: test complex ops ....................................................   Passed    1.27 sec
	        Start 163: test as_strided op
	163/251 Test #163: test as_strided op ..................................................   Passed    0.10 sec
	        Start 164: test scan op
	164/251 Test #164: test scan op ........................................................   Passed    0.08 sec
	        Start 165: test pad
	165/251 Test #165: test pad ............................................................   Passed    0.08 sec
	        Start 166: test power
	166/251 Test #166: test power ..........................................................   Passed    0.56 sec
	        Start 167: test where
	167/251 Test #167: test where ..........................................................   Passed    0.11 sec
	        Start 168: test stack
	168/251 Test #168: test stack ..........................................................   Passed    0.08 sec
	        Start 169: test full_like
	169/251 Test #169: test full_like ......................................................   Passed    0.10 sec
	        Start 170: test eye
	170/251 Test #170: test eye ............................................................   Passed    0.11 sec
	        Start 171: test tri
	171/251 Test #171: test tri ............................................................   Passed    0.09 sec
	        Start 172: test tril
	172/251 Test #172: test tril ...........................................................   Passed    0.08 sec
	        Start 173: test triu
	173/251 Test #173: test triu ...........................................................   Passed    0.06 sec
	        Start 174: test identity
	174/251 Test #174: test identity .......................................................   Passed    0.06 sec
	        Start 175: test eye with positive k offset
	175/251 Test #175: test eye with positive k offset .....................................   Passed    0.07 sec
	        Start 176: test eye with negative k offset
	176/251 Test #176: test eye with negative k offset .....................................   Passed    0.08 sec
	        Start 177: test basic clipping
	177/251 Test #177: test basic clipping .................................................   Passed    0.30 sec
	        Start 178: test clipping with only min
	178/251 Test #178: test clipping with only min .........................................   Passed    0.07 sec
	        Start 179: test clipping with only max
	179/251 Test #179: test clipping with only max .........................................   Passed    0.26 sec
	        Start 180: test linspace
	180/251 Test #180: test linspace .......................................................   Passed    0.10 sec
	        Start 181: test quantize dequantize
	181/251 Test #181: test quantize dequantize ............................................   Passed    0.08 sec
	        Start 182: test repeat
	182/251 Test #182: test repeat .........................................................   Passed    0.11 sec
	        Start 183: tile
	183/251 Test #183: tile ................................................................   Passed    0.06 sec
	        Start 184: tensordot
	184/251 Test #184: tensordot ...........................................................   Passed    0.11 sec
	        Start 185: outer
	185/251 Test #185: outer ...............................................................   Passed    0.11 sec
	        Start 186: inner
	186/251 Test #186: inner ...............................................................   Passed    0.12 sec
	        Start 187: test divmod
	187/251 Test #187: test divmod .........................................................   Passed    0.11 sec
	        Start 188: test diagonal
	188/251 Test #188: test diagonal .......................................................   Passed    0.13 sec
	        Start 189: test diag
	189/251 Test #189: test diag ...........................................................   Passed    0.12 sec
	        Start 190: test issubdtype
	190/251 Test #190: test issubdtype .....................................................   Passed    0.06 sec
	        Start 191: test atleast_1d
	191/251 Test #191: test atleast_1d .....................................................   Passed    0.06 sec
	        Start 192: test atleast_1d vector
	192/251 Test #192: test atleast_1d vector ..............................................   Passed    0.06 sec
	        Start 193: test atleast_2d
	193/251 Test #193: test atleast_2d .....................................................   Passed    0.06 sec
	        Start 194: test atleast_2d vector
	194/251 Test #194: test atleast_2d vector ..............................................   Passed    0.06 sec
	        Start 195: test atleast_3d
	195/251 Test #195: test atleast_3d .....................................................   Passed    0.04 sec
	        Start 196: test atleast_3d vector
	196/251 Test #196: test atleast_3d vector ..............................................   Passed    0.04 sec
	        Start 197: test topk
	197/251 Test #197: test topk ...........................................................   Passed    0.05 sec
	        Start 198: test meshgrid
	198/251 Test #198: test meshgrid .......................................................   Passed    0.09 sec
	        Start 199: test conv1d
	199/251 Test #199: test conv1d .........................................................   Passed    0.10 sec
	        Start 200: test conv2d
	200/251 Test #200: test conv2d .........................................................   Passed    0.12 sec
	        Start 201: test trace
	201/251 Test #201: test trace ..........................................................   Passed    0.11 sec
	        Start 202: test view
	202/251 Test #202: test view ...........................................................   Passed    0.06 sec
	        Start 203: test roll
	203/251 Test #203: test roll ...........................................................   Passed    0.06 sec
	        Start 204: test contiguous
	204/251 Test #204: test contiguous .....................................................   Passed    0.05 sec
	        Start 205: test bitwise shift operations
	205/251 Test #205: test bitwise shift operations .......................................   Passed    0.13 sec
	        Start 206: test conv_transpose1d with output_padding
	206/251 Test #206: test conv_transpose1d with output_padding ...........................   Passed    0.05 sec
	        Start 207: test conv_transpose2d with output_padding
	207/251 Test #207: test conv_transpose2d with output_padding ...........................   Passed    0.05 sec
	        Start 208: test conv_transpose3d with output_padding
	208/251 Test #208: test conv_transpose3d with output_padding ...........................   Passed    0.07 sec
	        Start 209: test fp8 conversion
	209/251 Test #209: test fp8 conversion .................................................   Passed    0.10 sec
	        Start 210: test max min with nan
	210/251 Test #210: test max min with nan ...............................................   Passed    0.05 sec
	        Start 211: test random key
	211/251 Test #211: test random key .....................................................   Passed    0.09 sec
	        Start 212: test global rng
	212/251 Test #212: test global rng .....................................................   Passed    0.06 sec
	        Start 213: test random split
	213/251 Test #213: test random split ...................................................   Passed    0.05 sec
	        Start 214: test random bits
	214/251 Test #214: test random bits ....................................................   Passed    0.75 sec
	        Start 215: test random uniform
	215/251 Test #215: test random uniform .................................................   Passed    0.82 sec
	        Start 216: test random normal
	216/251 Test #216: test random normal ..................................................   Passed    0.69 sec
	        Start 217: test random multivariate_normal
	217/251 Test #217: test random multivariate_normal .....................................   Passed    0.06 sec
	        Start 218: test random randint
	218/251 Test #218: test random randint .................................................   Passed    0.60 sec
	        Start 219: test random bernoulli
	219/251 Test #219: test random bernoulli ...............................................   Passed    0.08 sec
	        Start 220: Test truncated normal
	220/251 Test #220: Test truncated normal ...............................................   Passed    0.43 sec
	        Start 221: test categorical
	221/251 Test #221: test categorical ....................................................   Passed    0.76 sec
	        Start 222: test laplace
	222/251 Test #222: test laplace ........................................................   Passed    0.90 sec
	        Start 223: test stream management
	223/251 Test #223: test stream management ..............................................   Passed    0.06 sec
	        Start 224: test default stream in threads
	224/251 Test #224: test default stream in threads ......................................   Passed    0.06 sec
	        Start 225: test access stream in other thread
	225/251 Test #225: test access stream in other thread ..................................   Passed    0.07 sec
	        Start 226: test new stream in threads
	226/251 Test #226: test new stream in threads ..........................................   Passed    0.06 sec
	        Start 227: test thread local stream
	227/251 Test #227: test thread local stream ............................................   Passed    0.09 sec
	        Start 228: test get streams
	228/251 Test #228: test get streams ....................................................   Passed    0.05 sec
	        Start 229: test asynchronous launch
	229/251 Test #229: test asynchronous launch ............................................   Passed    0.06 sec
	        Start 230: test stream placement
	230/251 Test #230: test stream placement ...............................................   Passed    0.05 sec
	        Start 231: test scheduler races
	231/251 Test #231: test scheduler races ................................................   Passed    0.88 sec
	        Start 232: test type promotion
	232/251 Test #232: test type promotion .................................................   Passed    0.07 sec
	        Start 233: test normalize axis
	233/251 Test #233: test normalize axis .................................................   Passed    0.06 sec
	        Start 234: test finfo
	234/251 Test #234: test finfo ..........................................................   Passed    0.06 sec
	        Start 235: test iinfo
	235/251 Test #235: test iinfo ..........................................................   Passed    0.06 sec
	        Start 236: test simple vmap
	236/251 Test #236: test simple vmap ....................................................   Passed    0.18 sec
	        Start 237: test vmap with eval
	237/251 Test #237: test vmap with eval .................................................   Passed    0.09 sec
	        Start 238: test vmap comparison ops
	238/251 Test #238: test vmap comparison ops ............................................   Passed    0.08 sec
	        Start 239: test vmap creation ops
	239/251 Test #239: test vmap creation ops ..............................................   Passed    0.11 sec
	        Start 240: test vmap slice
	240/251 Test #240: test vmap slice .....................................................   Passed    0.11 sec
	        Start 241: test vmap concatenate
	241/251 Test #241: test vmap concatenate ...............................................   Passed    0.10 sec
	        Start 242: test vmap gather
	242/251 Test #242: test vmap gather ....................................................   Passed    0.05 sec
	        Start 243: test vmap take_along_axis with unmapped input and mapped index
	243/251 Test #243: test vmap take_along_axis with unmapped input and mapped index ......   Passed    0.04 sec
	        Start 244: test vmap scatter
	244/251 Test #244: test vmap scatter ...................................................   Passed    0.11 sec
	        Start 245: test vmap SVD
	245/251 Test #245: test vmap SVD .......................................................   Passed    0.06 sec
	        Start 246: test vmap dynamic slices
	246/251 Test #246: test vmap dynamic slices ............................................   Passed    0.06 sec
	        Start 247: test vmap floor_divide integer
	247/251 Test #247: test vmap floor_divide integer ......................................   Passed    0.07 sec
	        Start 248: test vulkan complex scalar view multiply regression
	248/251 Test #248: test vulkan complex scalar view multiply regression .................   Passed    0.04 sec
	        Start 249: test vulkan complex abs general layout regression
	249/251 Test #249: test vulkan complex abs general layout regression ...................   Passed    0.04 sec
	        Start 250: tests
	250/251 Test #250: tests ...............................................................   Passed   30.78 sec
	        Start 251: teardown
	251/251 Test #251: teardown ............................................................   Passed    1.07 sec
	
	100% tests passed, 0 tests failed out of 251
	
	Total Test time (real) =  81.88 sec


---
2026.7.26

One error case in test scatter types.

	Linear algebra related to AnyMagma is fixed.
	

	
	99% tests passed, 3 tests failed out of 237
	
	Total Test time (real) =  75.29 sec
	
	The following tests FAILED:
		147 - test scatter types (Failed)
		236 - tests (Failed)
		237 - teardown (Not Run)



---
2026.7.25

99% tests passed, 1 tests failed out of 222

Total Test time (real) = 113.21 sec

The following tests FAILED:

	222 - teardown (Not Run)

So parity tests are almost done, except for: 

    some fallbacks on strange cases of scattering and fp8 conversion
	linear algebra related to AnyMagma (shouldn't be a problem for all have been done with fixing CLMagma)
	
	
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
