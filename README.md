# MLX OpenCL Backend – Progress & Features

This document summarizes the current state of the MLX OpenCL backend, highlighting key features, implemented primitives, and the overall progress. The backend aims to provide a complete, high‑performance OpenCL implementation of MLX’s core operations, with a focus on portability and distributed training.

---
2026.7.26

One error case in test scatter types.

	Linear algebra related to AnyMagma is fixed.
	
	(base) jc@U1:~/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests$ make test -j16
	Running tests...
	Test project /home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests
	Connected to MAKE jobserver
	        Start   1: test simple allocations
	  1/237 Test   #1: test simple allocations .............................................   Passed    0.02 sec
	        Start   2: test large allocations
	  2/237 Test   #2: test large allocations ..............................................   Passed   11.15 sec
	        Start   3: test arg reduce small
	  3/237 Test   #3: test arg reduce small ...............................................   Passed    0.04 sec
	        Start   4: test arg reduce against cpu
	  4/237 Test   #4: test arg reduce against cpu .........................................   Passed    0.02 sec
	        Start   5: test arg reduce bool
	  5/237 Test   #5: test arg reduce bool ................................................   Passed    0.04 sec
	        Start   6: test arg reduce edge cases
	  6/237 Test   #6: test arg reduce edge cases ..........................................   Passed    0.03 sec
	        Start   7: test arg reduce irregular strides
	  7/237 Test   #7: test arg reduce irregular strides ...................................   Passed    0.03 sec
	        Start   8: test array basics
	  8/237 Test   #8: test array basics ...................................................   Passed    0.06 sec
	        Start   9: test array types
	  9/237 Test   #9: test array types ....................................................   Passed    0.05 sec
	        Start  10: test array metadata
	 10/237 Test  #10: test array metadata .................................................   Passed    0.04 sec
	        Start  11: test array iteration
	 11/237 Test  #11: test array iteration ................................................   Passed    0.03 sec
	        Start  12: test array shared buffer
	 12/237 Test  #12: test array shared buffer ............................................   Passed    0.03 sec
	        Start  13: test make empty array
	 13/237 Test  #13: test make empty array ...............................................   Passed    0.04 sec
	        Start  14: test make array from user buffer
	 14/237 Test  #14: test make array from user buffer ....................................   Passed    0.03 sec
	        Start  15: test negative indexing for shape/strides
	 15/237 Test  #15: test negative indexing for shape/strides ............................   Passed    0.03 sec
	        Start  16: test stop gradient
	 16/237 Test  #16: test stop gradient ..................................................   Passed    0.04 sec
	        Start  17: test jvp
	 17/237 Test  #17: test jvp ............................................................   Passed    0.08 sec
	        Start  18: test vjp
	 18/237 Test  #18: test vjp ............................................................   Passed    0.08 sec
	        Start  19: test grad
	 19/237 Test  #19: test grad ...........................................................   Passed    0.07 sec
	        Start  20: test transform container reuse does not accumulate stale wrappers
	 20/237 Test  #20: test transform container reuse does not accumulate stale wrappers ...   Passed    0.04 sec
	        Start  21: test creation grads
	 21/237 Test  #21: test creation grads .................................................   Passed    0.03 sec
	        Start  22: test op vjps
	 22/237 Test  #22: test op vjps ........................................................   Passed    1.33 sec
	        Start  23: test gather and take grads
	 23/237 Test  #23: test gather and take grads ..........................................   Passed    0.73 sec
	        Start  24: test slice grads
	 24/237 Test  #24: test slice grads ....................................................   Passed    0.31 sec
	        Start  25: test min and max vjp
	 25/237 Test  #25: test min and max vjp ................................................   Passed    0.07 sec
	        Start  26: test reshape and transpose grads
	 26/237 Test  #26: test reshape and transpose grads ....................................   Passed    0.07 sec
	        Start  27: test copy grads
	 27/237 Test  #27: test copy grads .....................................................   Passed    0.05 sec
	        Start  28: test matmul vjp
	 28/237 Test  #28: test matmul vjp .....................................................   Passed    0.06 sec
	        Start  29: test concatenate grads
	 29/237 Test  #29: test concatenate grads ..............................................   Passed    0.06 sec
	        Start  30: test split grads
	 30/237 Test  #30: test split grads ....................................................   Passed    0.04 sec
	        Start  31: test comparison grads
	 31/237 Test  #31: test comparison grads ...............................................   Passed    0.10 sec
	        Start  32: test as_strided grads
	 32/237 Test  #32: test as_strided grads ...............................................   Passed    0.08 sec
	        Start  33: test jvp from vjp
	 33/237 Test  #33: test jvp from vjp ...................................................   Passed    0.56 sec
	        Start  34: test complex gradients
	 34/237 Test  #34: test complex gradients ..............................................   Passed    0.31 sec
	        Start  35: test scan grads
	 35/237 Test  #35: test scan grads .....................................................   Passed    0.09 sec
	        Start  36: test update state
	 36/237 Test  #36: test update state ...................................................   Passed    0.04 sec
	        Start  37: test grad types
	 37/237 Test  #37: test grad types .....................................................   Passed    0.03 sec
	        Start  38: test grad dynamic slices
	 38/237 Test  #38: test grad dynamic slices ............................................   Passed    0.04 sec
	        Start  39: test masked_scatter autograd
	 39/237 Test  #39: test masked_scatter autograd ........................................   Passed    0.05 sec
	        Start  40: test matmul
	 40/237 Test  #40: test matmul .........................................................   Passed    0.09 sec
	        Start  41: test simple compile
	 41/237 Test  #41: test simple compile .................................................   Passed    0.07 sec
	        Start  42: test compile with grad
	 42/237 Test  #42: test compile with grad ..............................................   Passed    0.05 sec
	        Start  43: test compile inputs with primitive
	 43/237 Test  #43: test compile inputs with primitive ..................................   Passed    0.08 sec
	        Start  44: test compile with created array
	 44/237 Test  #44: test compile with created array .....................................   Passed    0.03 sec
	        Start  45: test nested compile
	 45/237 Test  #45: test nested compile .................................................   Passed    0.05 sec
	        Start  46: test enable and disable compile
	 46/237 Test  #46: test enable and disable compile .....................................   Passed    0.04 sec
	        Start  47: test simplify scalars
	 47/237 Test  #47: test simplify scalars ...............................................   Passed    0.04 sec
	        Start  48: test simplify
	 48/237 Test  #48: test simplify .......................................................   Passed    0.04 sec
	        Start  49: test simplify noops
	 49/237 Test  #49: test simplify noops .................................................   Passed    0.04 sec
	        Start  50: test no simplify
	 50/237 Test  #50: test no simplify ....................................................   Passed    0.02 sec
	        Start  51: test simplify multi output
	 51/237 Test  #51: test simplify multi output ..........................................   Passed    0.06 sec
	        Start  52: test compile unary fused
	 52/237 Test  #52: test compile unary fused ............................................   Passed    0.08 sec
	        Start  53: test compile binary fused
	 53/237 Test  #53: test compile binary fused ...........................................   Passed    0.04 sec
	        Start  54: test compile gelu
	 54/237 Test  #54: test compile gelu ...................................................   Passed    0.08 sec
	        Start  55: test compile tape with outside parents
	 55/237 Test  #55: test compile tape with outside parents ..............................   Passed    0.07 sec
	        Start  56: test compile across streams
	 56/237 Test  #56: test compile across streams .........................................   Passed    0.04 sec
	        Start  57: test compile deep graph
	 57/237 Test  #57: test compile deep graph .............................................   Passed    0.07 sec
	        Start  58: test compile repeat input
	 58/237 Test  #58: test compile repeat input ...........................................   Passed    0.08 sec
	        Start  59: test compile compiled function
	 59/237 Test  #59: test compile compiled function ......................................   Passed    0.03 sec
	        Start  60: test transform compiled function
	 60/237 Test  #60: test transform compiled function ....................................   Passed    0.04 sec
	        Start  61: test fusion kernel reuse
	 61/237 Test  #61: test fusion kernel reuse ............................................   Passed    0.04 sec
	        Start  62: test fusion types
	 62/237 Test  #62: test fusion types ...................................................   Passed    0.03 sec
	        Start  63: test shapeless compile
	 63/237 Test  #63: test shapeless compile ..............................................   Passed    0.03 sec
	        Start  64: test compile strides
	 64/237 Test  #64: test compile strides ................................................   Passed    0.06 sec
	        Start  65: test compile change streams
	 65/237 Test  #65: test compile change streams .........................................   Passed    0.04 sec
	        Start  66: test compile lambda
	 66/237 Test  #66: test compile lambda .................................................   Passed    0.05 sec
	        Start  67: test compile with no-ops
	 67/237 Test  #67: test compile with no-ops ............................................   Passed    0.04 sec
	        Start  68: test compile random bits
	 68/237 Test  #68: test compile random bits ............................................   Passed    0.03 sec
	        Start  69: test arange
	 69/237 Test  #69: test arange .........................................................   Passed    0.08 sec
	        Start  70: test astype
	 70/237 Test  #70: test astype .........................................................   Passed    0.05 sec
	        Start  71: test full
	 71/237 Test  #71: test full ...........................................................   Passed    0.07 sec
	        Start  72: test simple custom vjp
	 72/237 Test  #72: test simple custom vjp ..............................................   Passed    0.04 sec
	        Start  73: test checkpointing
	 73/237 Test  #73: test checkpointing ..................................................   Passed    0.18 sec
	        Start  74: test device placement
	 74/237 Test  #74: test device placement ...............................................   Passed    0.04 sec
	        Start  75: test einsum path
	 75/237 Test  #75: test einsum path ....................................................   Passed    0.05 sec
	        Start  76: test einsum
	 76/237 Test  #76: test einsum .........................................................   Passed    0.07 sec
	        Start  77: test eval
	 77/237 Test  #77: test eval ...........................................................   Passed    0.04 sec
	        Start  78: test eval multiple
	 78/237 Test  #78: test eval multiple ..................................................   Passed    0.04 sec
	        Start  79: test eval with tracer when not tracing
	 79/237 Test  #79: test eval with tracer when not tracing ..............................   Passed    0.05 sec
	        Start  80: test eval graph retention when not tracing
	 80/237 Test  #80: test eval graph retention when not tracing ..........................   Passed    0.05 sec
	        Start  81: test export basic functions
	 81/237 Test  #81: test export basic functions .........................................   Passed    0.05 sec
	        Start  82: test export function with no inputs
	 82/237 Test  #82: test export function with no inputs .................................   Passed    0.05 sec
	        Start  83: test export multi output primitives
	 83/237 Test  #83: test export multi output primitives .................................   Passed    0.04 sec
	        Start  84: test export primitives with state
	 84/237 Test  #84: test export primitives with state ...................................   Passed    0.03 sec
	        Start  85: test export functions with kwargs
	 85/237 Test  #85: test export functions with kwargs ...................................   Passed    0.05 sec
	        Start  86: test export function with variable inputs
	 86/237 Test  #86: test export function with variable inputs ...........................   Passed    0.04 sec
	        Start  87: test export function on different stream
	 87/237 Test  #87: test export function on different stream ............................   Passed    0.02 sec
	        Start  88: test fft basics
	 88/237 Test  #88: test fft basics .....................................................   Passed    0.94 sec
	        Start  89: test real ffts
	 89/237 Test  #89: test real ffts ......................................................   Passed    0.18 sec
	        Start  90: test fftn
	 90/237 Test  #90: test fftn ...........................................................   Passed    0.54 sec
	        Start  91: test fft with provided shape
	 91/237 Test  #91: test fft with provided shape ........................................   Passed    0.04 sec
	        Start  92: test fft vmap
	 92/237 Test  #92: test fft vmap .......................................................   Passed    0.49 sec
	        Start  93: test fft grads
	 93/237 Test  #93: test fft grads ......................................................   Passed    0.79 sec
	        Start  94: test fftshift and ifftshift
	 94/237 Test  #94: test fftshift and ifftshift .........................................   Passed    0.06 sec
	        Start  95: [mlx.core.linalg.norm] no ord
	 95/237 Test  #95: [mlx.core.linalg.norm] no ord .......................................   Passed    0.19 sec
	        Start  96: [mlx.core.linalg.norm] double ord
	 96/237 Test  #96: [mlx.core.linalg.norm] double ord ...................................   Passed    0.85 sec
	        Start  97: [mlx.core.linalg.norm] string ord
	 97/237 Test  #97: [mlx.core.linalg.norm] string ord ...................................   Passed    1.02 sec
	        Start  98: test QR factorization
	 98/237 Test  #98: test QR factorization ...............................................   Passed    0.08 sec
	        Start  99: test SVD factorization
	 99/237 Test  #99: test SVD factorization ..............................................   Passed    0.47 sec
	        Start 100: test matrix inversion
	100/237 Test #100: test matrix inversion ...............................................   Passed    0.09 sec
	        Start 101: test matrix cholesky
	101/237 Test #101: test matrix cholesky ................................................   Passed    0.06 sec
	        Start 102: test matrix pseudo-inverse
	102/237 Test #102: test matrix pseudo-inverse ..........................................   Passed    0.13 sec
	        Start 103: test cross product
	103/237 Test #103: test cross product ..................................................   Passed    0.10 sec
	        Start 104: test matrix eigh
	104/237 Test #104: test matrix eigh ....................................................   Passed    0.09 sec
	        Start 105: test lu
	105/237 Test #105: test lu .............................................................   Passed    0.43 sec
	        Start 106: test solve
	106/237 Test #106: test solve ..........................................................   Passed    0.10 sec
	        Start 107: test solve_triangluar
	107/237 Test #107: test solve_triangluar ...............................................   Passed    0.07 sec
	        Start 108: test det
	108/237 Test #108: test det ............................................................   Passed    0.04 sec
	        Start 109: test slogdet
	109/237 Test #109: test slogdet ........................................................   Passed    0.04 sec
	        Start 110: test save_safetensors
	110/237 Test #110: test save_safetensors ...............................................   Passed    0.03 sec
	        Start 111: test safetensors file boundary validation
	111/237 Test #111: test safetensors file boundary validation ...........................   Passed    0.05 sec
	        Start 112: test gguf
	112/237 Test #112: test gguf ...........................................................   Passed    0.05 sec
	        Start 113: test gguf metadata
	113/237 Test #113: test gguf metadata ..................................................   Passed    0.06 sec
	        Start 114: test single array serialization
	114/237 Test #114: test single array serialization .....................................   Passed    0.07 sec
	        Start 115: test copy
	115/237 Test #115: test copy ...........................................................   Passed    0.04 sec
	        Start 116: test reshape
	116/237 Test #116: test reshape ........................................................   Passed    0.02 sec
	        Start 117: test flatten
	117/237 Test #117: test flatten ........................................................   Passed    0.03 sec
	        Start 118: test unflatten
	118/237 Test #118: test unflatten ......................................................   Passed    0.02 sec
	        Start 119: test squeeze and expand
	119/237 Test #119: test squeeze and expand .............................................   Passed    0.04 sec
	        Start 120: test slice
	120/237 Test #120: test slice ..........................................................   Passed    0.07 sec
	        Start 121: test slice update
	121/237 Test #121: test slice update ...................................................   Passed    0.03 sec
	        Start 122: test slice update add
	122/237 Test #122: test slice update add ...............................................   Passed    0.06 sec
	        Start 123: test dynamic slice
	123/237 Test #123: test dynamic slice ..................................................   Passed    0.04 sec
	        Start 124: test dynamic slice update
	124/237 Test #124: test dynamic slice update ...........................................   Passed    0.03 sec
	        Start 125: test split
	125/237 Test #125: test split ..........................................................   Passed    0.03 sec
	        Start 126: test swap and move axes
	126/237 Test #126: test swap and move axes .............................................   Passed    0.03 sec
	        Start 127: test transpose
	127/237 Test #127: test transpose ......................................................   Passed    0.04 sec
	        Start 128: test comparison ops
	128/237 Test #128: test comparison ops .................................................   Passed    0.11 sec
	        Start 129: test is nan
	129/237 Test #129: test is nan .........................................................   Passed    0.07 sec
	        Start 130: test is inf
	130/237 Test #130: test is inf .........................................................   Passed    0.06 sec
	        Start 131: test all close
	131/237 Test #131: test all close ......................................................   Passed    0.11 sec
	        Start 132: test is close
	132/237 Test #132: test is close .......................................................   Passed    0.06 sec
	        Start 133: test reduction ops
	133/237 Test #133: test reduction ops ..................................................   Passed    0.26 sec
	        Start 134: test irregular binary ops
	134/237 Test #134: test irregular binary ops ...........................................   Passed    0.07 sec
	        Start 135: test arithmetic unary ops
	135/237 Test #135: test arithmetic unary ops ...........................................   Passed    0.37 sec
	        Start 136: test error functions
	136/237 Test #136: test error functions ................................................   Passed    0.04 sec
	        Start 137: test arithmetic binary ops
	137/237 Test #137: test arithmetic binary ops ..........................................   Passed    0.23 sec
	        Start 138: test broadcast
	138/237 Test #138: test broadcast ......................................................   Passed    0.06 sec
	        Start 139: test gather
	139/237 Test #139: test gather .........................................................   Passed    0.41 sec
	        Start 140: test take
	140/237 Test #140: test take ...........................................................   Passed    1.71 sec
	        Start 141: test take along axis
	141/237 Test #141: test take along axis ................................................   Passed    1.33 sec
	        Start 142: test put along axis
	142/237 Test #142: test put along axis .................................................   Passed    0.06 sec
	        Start 143: test scatter
	143/237 Test #143: test scatter ........................................................   Passed    0.22 sec
	        Start 144: test masked_scatter
	144/237 Test #144: test masked_scatter .................................................   Passed    0.06 sec
	        Start 145: test is positive infinity
	145/237 Test #145: test is positive infinity ...........................................   Passed    0.07 sec
	        Start 146: test is negative infinity
	146/237 Test #146: test is negative infinity ...........................................   Passed    0.07 sec
	        Start 147: test scatter types
	147/237 Test #147: test scatter types ..................................................***Failed    0.09 sec
	        Start 148: test complex ops
	148/237 Test #148: test complex ops ....................................................   Passed    0.93 sec
	        Start 149: test as_strided op
	149/237 Test #149: test as_strided op ..................................................   Passed    0.04 sec
	        Start 150: test scan op
	150/237 Test #150: test scan op ........................................................   Passed    0.04 sec
	        Start 151: test pad
	151/237 Test #151: test pad ............................................................   Passed    0.05 sec
	        Start 152: test power
	152/237 Test #152: test power ..........................................................   Passed    0.93 sec
	        Start 153: test where
	153/237 Test #153: test where ..........................................................   Passed    0.05 sec
	        Start 154: test stack
	154/237 Test #154: test stack ..........................................................   Passed    0.04 sec
	        Start 155: test full_like
	155/237 Test #155: test full_like ......................................................   Passed    0.06 sec
	        Start 156: test eye
	156/237 Test #156: test eye ............................................................   Passed    0.07 sec
	        Start 157: test tri
	157/237 Test #157: test tri ............................................................   Passed    0.03 sec
	        Start 158: test tril
	158/237 Test #158: test tril ...........................................................   Passed    0.05 sec
	        Start 159: test triu
	159/237 Test #159: test triu ...........................................................   Passed    0.06 sec
	        Start 160: test identity
	160/237 Test #160: test identity .......................................................   Passed    0.04 sec
	        Start 161: test eye with positive k offset
	161/237 Test #161: test eye with positive k offset .....................................   Passed    0.07 sec
	        Start 162: test eye with negative k offset
	162/237 Test #162: test eye with negative k offset .....................................   Passed    0.03 sec
	        Start 163: test basic clipping
	163/237 Test #163: test basic clipping .................................................   Passed    0.06 sec
	        Start 164: test clipping with only min
	164/237 Test #164: test clipping with only min .........................................   Passed    0.03 sec
	        Start 165: test clipping with only max
	165/237 Test #165: test clipping with only max .........................................   Passed    0.06 sec
	        Start 166: test linspace
	166/237 Test #166: test linspace .......................................................   Passed    0.09 sec
	        Start 167: test quantize dequantize
	167/237 Test #167: test quantize dequantize ............................................   Passed    0.04 sec
	        Start 168: test repeat
	168/237 Test #168: test repeat .........................................................   Passed    0.07 sec
	        Start 169: tile
	169/237 Test #169: tile ................................................................   Passed    0.08 sec
	        Start 170: tensordot
	170/237 Test #170: tensordot ...........................................................   Passed    0.07 sec
	        Start 171: outer
	171/237 Test #171: outer ...............................................................   Passed    0.05 sec
	        Start 172: inner
	172/237 Test #172: inner ...............................................................   Passed    0.05 sec
	        Start 173: test divmod
	173/237 Test #173: test divmod .........................................................   Passed    0.04 sec
	        Start 174: test diagonal
	174/237 Test #174: test diagonal .......................................................   Passed    0.09 sec
	        Start 175: test diag
	175/237 Test #175: test diag ...........................................................   Passed    0.07 sec
	        Start 176: test issubdtype
	176/237 Test #176: test issubdtype .....................................................   Passed    0.04 sec
	        Start 177: test atleast_1d
	177/237 Test #177: test atleast_1d .....................................................   Passed    0.03 sec
	        Start 178: test atleast_1d vector
	178/237 Test #178: test atleast_1d vector ..............................................   Passed    0.05 sec
	        Start 179: test atleast_2d
	179/237 Test #179: test atleast_2d .....................................................   Passed    0.04 sec
	        Start 180: test atleast_2d vector
	180/237 Test #180: test atleast_2d vector ..............................................   Passed    0.04 sec
	        Start 181: test atleast_3d
	181/237 Test #181: test atleast_3d .....................................................   Passed    0.04 sec
	        Start 182: test atleast_3d vector
	182/237 Test #182: test atleast_3d vector ..............................................   Passed    0.04 sec
	        Start 183: test topk
	183/237 Test #183: test topk ...........................................................   Passed    0.06 sec
	        Start 184: test meshgrid
	184/237 Test #184: test meshgrid .......................................................   Passed    0.05 sec
	        Start 185: test conv1d
	185/237 Test #185: test conv1d .........................................................   Passed    0.10 sec
	        Start 186: test conv2d
	186/237 Test #186: test conv2d .........................................................   Passed    0.10 sec
	        Start 187: test trace
	187/237 Test #187: test trace ..........................................................   Passed    0.08 sec
	        Start 188: test view
	188/237 Test #188: test view ...........................................................   Passed    0.06 sec
	        Start 189: test roll
	189/237 Test #189: test roll ...........................................................   Passed    0.08 sec
	        Start 190: test contiguous
	190/237 Test #190: test contiguous .....................................................   Passed    0.05 sec
	        Start 191: test bitwise shift operations
	191/237 Test #191: test bitwise shift operations .......................................   Passed    2.52 sec
	        Start 192: test conv_transpose1d with output_padding
	192/237 Test #192: test conv_transpose1d with output_padding ...........................   Passed    0.05 sec
	        Start 193: test conv_transpose2d with output_padding
	193/237 Test #193: test conv_transpose2d with output_padding ...........................   Passed    0.05 sec
	        Start 194: test conv_transpose3d with output_padding
	194/237 Test #194: test conv_transpose3d with output_padding ...........................   Passed    0.05 sec
	        Start 195: test fp8 conversion
	195/237 Test #195: test fp8 conversion .................................................   Passed    0.06 sec
	        Start 196: test max min with nan
	196/237 Test #196: test max min with nan ...............................................   Passed    0.07 sec
	        Start 197: test random key
	197/237 Test #197: test random key .....................................................   Passed    0.05 sec
	        Start 198: test global rng
	198/237 Test #198: test global rng .....................................................   Passed    0.03 sec
	        Start 199: test random split
	199/237 Test #199: test random split ...................................................   Passed    0.03 sec
	        Start 200: test random bits
	200/237 Test #200: test random bits ....................................................   Passed    0.59 sec
	        Start 201: test random uniform
	201/237 Test #201: test random uniform .................................................   Passed    0.08 sec
	        Start 202: test random normal
	202/237 Test #202: test random normal ..................................................   Passed    0.08 sec
	        Start 203: test random multivariate_normal
	203/237 Test #203: test random multivariate_normal .....................................   Passed    0.05 sec
	        Start 204: test random randint
	204/237 Test #204: test random randint .................................................   Passed    0.16 sec
	        Start 205: test random bernoulli
	205/237 Test #205: test random bernoulli ...............................................   Passed    0.06 sec
	        Start 206: Test truncated normal
	206/237 Test #206: Test truncated normal ...............................................   Passed    0.08 sec
	        Start 207: test categorical
	207/237 Test #207: test categorical ....................................................   Passed    0.08 sec
	        Start 208: test laplace
	208/237 Test #208: test laplace ........................................................   Passed    0.29 sec
	        Start 209: test stream management
	209/237 Test #209: test stream management ..............................................   Passed    0.04 sec
	        Start 210: test default stream in threads
	210/237 Test #210: test default stream in threads ......................................   Passed    0.03 sec
	        Start 211: test access stream in other thread
	211/237 Test #211: test access stream in other thread ..................................   Passed    0.03 sec
	        Start 212: test new stream in threads
	212/237 Test #212: test new stream in threads ..........................................   Passed    0.03 sec
	        Start 213: test thread local stream
	213/237 Test #213: test thread local stream ............................................   Passed    0.06 sec
	        Start 214: test get streams
	214/237 Test #214: test get streams ....................................................   Passed    0.04 sec
	        Start 215: test asynchronous launch
	215/237 Test #215: test asynchronous launch ............................................   Passed    0.03 sec
	        Start 216: test stream placement
	216/237 Test #216: test stream placement ...............................................   Passed    0.03 sec
	        Start 217: test scheduler races
	217/237 Test #217: test scheduler races ................................................   Passed    1.00 sec
	        Start 218: test type promotion
	218/237 Test #218: test type promotion .................................................   Passed    0.05 sec
	        Start 219: test normalize axis
	219/237 Test #219: test normalize axis .................................................   Passed    0.05 sec
	        Start 220: test finfo
	220/237 Test #220: test finfo ..........................................................   Passed    0.04 sec
	        Start 221: test iinfo
	221/237 Test #221: test iinfo ..........................................................   Passed    0.04 sec
	        Start 222: test simple vmap
	222/237 Test #222: test simple vmap ....................................................   Passed    0.11 sec
	        Start 223: test vmap with eval
	223/237 Test #223: test vmap with eval .................................................   Passed    0.04 sec
	        Start 224: test vmap comparison ops
	224/237 Test #224: test vmap comparison ops ............................................   Passed    0.05 sec
	        Start 225: test vmap creation ops
	225/237 Test #225: test vmap creation ops ..............................................   Passed    0.09 sec
	        Start 226: test vmap slice
	226/237 Test #226: test vmap slice .....................................................   Passed    0.07 sec
	        Start 227: test vmap concatenate
	227/237 Test #227: test vmap concatenate ...............................................   Passed    0.11 sec
	        Start 228: test vmap gather
	228/237 Test #228: test vmap gather ....................................................   Passed    0.04 sec
	        Start 229: test vmap take_along_axis with unmapped input and mapped index
	229/237 Test #229: test vmap take_along_axis with unmapped input and mapped index ......   Passed    0.05 sec
	        Start 230: test vmap scatter
	230/237 Test #230: test vmap scatter ...................................................   Passed    0.10 sec
	        Start 231: test vmap SVD
	231/237 Test #231: test vmap SVD .......................................................   Passed    0.06 sec
	        Start 232: test vmap dynamic slices
	232/237 Test #232: test vmap dynamic slices ............................................   Passed    0.05 sec
	        Start 233: test vmap floor_divide integer
	233/237 Test #233: test vmap floor_divide integer ......................................   Passed    0.05 sec
	        Start 234: test vulkan complex scalar view multiply regression
	234/237 Test #234: test vulkan complex scalar view multiply regression .................   Passed    0.04 sec
	        Start 235: test vulkan complex abs general layout regression
	235/237 Test #235: test vulkan complex abs general layout regression ...................   Passed    0.07 sec
	        Start 236: tests
	236/237 Test #236: tests ...............................................................***Failed   32.87 sec
	        Start 237: teardown
	Could not find executable /home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/test_teardown
	Looked in the following places:
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Release/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Release/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Debug/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Debug/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/MinSizeRel/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/MinSizeRel/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/RelWithDebInfo/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/RelWithDebInfo/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Deployment/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Deployment/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Development/test_teardown
	/home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Development/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Release/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Release/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Debug/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Debug/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/MinSizeRel/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/MinSizeRel/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/RelWithDebInfo/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/RelWithDebInfo/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Deployment/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Deployment/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Development/test_teardown
	home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Development/test_teardown
	Unable to find executable: /home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/test_teardown
	237/237 Test #237: teardown ............................................................***Not Run   0.00 sec
	
	99% tests passed, 3 tests failed out of 237
	
	Total Test time (real) =  75.29 sec
	
	The following tests FAILED:
		147 - test scatter types (Failed)
		236 - tests (Failed)
		237 - teardown (Not Run)
	Errors while running CTest
	Output from these tests are in: /home/jc/Downloads/vdss/mlx-opencwl/mlx-opencl/build/tests/Testing/Temporary/LastTest.log
	Use "--rerun-failed --output-on-failure" to re-run the failed cases verbosely.
	make: *** [Makefile:71：test] 错误 8


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
