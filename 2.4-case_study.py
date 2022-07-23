import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import time

dtype = 'float32'
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)

time_start = time.time()
c_mm_relu = np.maximum(a_np @ b_np, 0)
print(f"mm_relu(numpy) cost {(time.time() - time_start):.3f}s")


# implement mm_relu
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
  Y = np.empty((128, 128), dtype="float32")
  for i in range(128):
    for j in range(128):
      Y[i, j] = 0
      for k in range(128):
        Y[i, j] += A[i, k] * B[k, j]
      C[i, j] = max(Y[i, j], 0)

c_np = np.zeros((128, 128), dtype=dtype)
time_start = time.time()
lnumpy_mm_relu(a_np, b_np, c_np)
print(f"lnumpy_mm_relu(handcraft) cost {(time.time() - time_start):.3f}s")

np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)


# implement mm_reluV2
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
  Y = np.empty((128, 128), dtype="float32")
  for i in range(128):
    for j0 in range(32):
      for k in range(128):
        for j1 in range(4):
          j = j0 * 4 + j1
          if k == 0:
            Y[i, j] = 0
          Y[i, j] = Y[i, j] + A[i, k] * B[k, j]

  for i in range(128):
    for j in range(128):
      C[i, j] = max(Y[i, j], 0)

c_mp = np.zeros((128, 128), dtype=dtype)
time_start = time.time()
lnumpy_mm_relu_v2(a_np, b_np, c_mp)
print(f"lnumpy_mm_relu_v2 cost {time.time() - time_start : .3f}s")
np.testing.assert_allclose(c_mm_relu, c_np, atol=1e-5)

# TensorIR
@tvm.script.ir_module
class MyModule:
  @T.prim_func
  def mm_relu(A: T.Buffer[(128, 128), "float32"],
              B: T.Buffer[(128, 128), "float32"],
              C: T.Buffer[(128, 128), "float32"]):
    T.func_attr({"global_symbol": "mm_relu", "tir_noalias": True})
    Y = T.alloc_buffer((128, 128), dtype="float32")
    for i, j, k in T.grid(128, 128, 128):
      with T.block("Y"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        with T.init():
          Y[vi, vj] = T.float32(0)
        Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

    for i, j in T.grid(128, 128):
      with T.block("C"):
        vi, vj = T.axis.remap("SS", [i, j])
        C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# import IPython
# IPython.display.Code(MyModule.script(), language="python")
print("\mold version of MyModule:")
print(MyModule.script())

sch = tvm.tir.Schedule(MyModule)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=(None, 4))
print("\nsplit j to [j0, j1] shc.mod:")
# print(MyModule.script())
print(sch.mod.script())

# reorder to [j0, k, j1]
sch.reorder(j0, k, j1)
print("\nafter reorder[j0, k, j1]:")
print(sch.mod.script())

# move block C to block Y
block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)
print("\nmove block C to block Y")
print(sch.mod.script())


# decompose reduction
sch.decompose_reduction(block_Y, k)
print("\ndecompose k of block_Y")
print(sch.mod.script())


# build and run old MyModule
rt_lib = tvm.build(MyModule, target="llvm")
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
# get runtime func
func_mm_relu = rt_lib['mm_relu']
time_start = time.time()
func_mm_relu(a_nd, b_nd, c_nd)
print(f"func_mm_relu(tvm) cost {time.time() - time_start:.3f}s")
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), atol=1e-5)

# transform version
rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# eval time cost
f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)


# try differenct jfactor
def transform(mod, jfactor):
  sch = tvm.tir.Schedule(mod)
  block_Y = sch.get_block("Y", func_name="mm_relu")
  i, j, k = sch.get_loops(block_Y)
  j0, j1 = sch.split(j, factors=[None, jfactor])
  sch.reorder(j0, k, j1)
  block_C = sch.get_block("C", func_name="mm_relu")
  sch.reverse_compute_at(block_C, j0)
  return sch.mod

jfactors = [4, 8, 16, 32, 64, 128, 256, 512]

for jfactor in jfactors:
  mod_transformed = transform(MyModule, jfactor=jfactor)
  rt_lib_transformed = tvm.build(mod_transformed, "llvm")
  f_timer_transformed = rt_lib_transformed.time_evaluator("mm_relu", tvm.cpu())
  print("Time cost of transformed mod_transformed(%d) %g sec" % (jfactor, f_timer_transformed(a_nd, b_nd, c_nd).mean))



# tensor expression
from tvm import te
A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")
te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
print(MyModuleFromTE.script())
