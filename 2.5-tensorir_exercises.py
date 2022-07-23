import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# high level sum
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
c_np = a + b
print(c_np)

# low level sum
def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
  for i in range(4):
    for j in range(4):
      c[i, j] = a[i, j] + b[i, j]

c_lnumpy = np.empty([4, 4], dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
print(c_lnumpy)

# tensorIR sum
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer[(4, 4), "int64"],
          B: T.Buffer[(4, 4), "int64"],
          C: T.Buffer[(4, 4), "int64"]):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty([4, 4], dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
print("test sum is good!")

# 1.2 broadcast
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
c_np = a + b
print(c_np)

@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer[(4, 4), "int64"],
          B: T.Buffer[(4), "int64"],
          C: T.Buffer[(4, 4), "int64"]):
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vj]

rt_lib = tvm.build(MyAdd, target="llvm")    
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-5)
print("test broadcast sum is good!")


# 2-d convolution
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

import torch
data_torch = torch.Tensor(data)
weight_torch =torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
print(conv_torch)

# 2- convolution (python)
c_np = np.empty(conv_torch.shape, dtype=np.int64)
print(c_np.shape)

for i in range(N):
  for j in range(CO):
    for h in range(OUT_H):
      for w in range(OUT_W):
        c_np[i, j, h, w] = 0
        for ci in range(CI):
          for k1 in range(K):
            for k2 in range(K):
              c_np[i, j, h, w] += data[i, ci, h+k1, w+k2] * weight[j, ci, k1, k2]
np.testing.assert_allclose(c_np, conv_torch, rtol=1e-5)
print("test 2d-conv(python) is good!")

# 2-d convolution (TensorIR)
@tvm.script.ir_module
class MyConv:
  @T.prim_func
  # def conv(A: T.Buffer[(N, CI, H, W), "int64"],
  #          B: T.Buffer[(CO, CI, K, K), "int64"],
  #          C: T.Buffer[(N, CO, OUT_H, OUT_W), "int64"]):
  def conv(A: T.Buffer[(N, CI, 8, 8), "int64"],
           B: T.Buffer[(CO, CI, K, K), "int64"],
           C: T.Buffer[(N, CO, OUT_H, OUT_W), "int64"]):           
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    for i, j, h, w, ci, k1, k2, in T.grid(N, CO, OUT_H, OUT_W, CI, K, K):
      with T.block("C"):
        vi, vj, vh, vw = T.axis.remap("SSSS", [i, j, h, w])
        vci, vk1, vk2 = T.axis.remap("SRR", [ci, k1, k2])
        with T.init():
          C[vi, vj, vh, vw] = T.int64(0)
        C[vi, vj, vh, vw] += A[vi, vci, vh+vk1, vw+vk2] * B[vj, vci, vk1, vk2]

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)
print('test 2d-conv(tensorIR) is good!')


# 2.1 parallel, unroll, vectorize
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer[(4, 4), "int64"],
          B: T.Buffer[(4, 4), "int64"],
          C: T.Buffer[(4, 4), "int64"]):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
print(sch.mod.script())


# 2.2 transform matmul
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
  Y = np.empty((16, 128, 128), dtype="float32")
  for n in range(16):
    for i in range(128):
      for j in range(128):
        for k in range(128):
          if k == 0:
              Y[n, i, j] = 0
          Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
  
  for n in range(16):
    for i in range(128):
      for j in range(128):
        C[n, i, j] = max(Y[n, i, j], 0)


# tensorIR
# @tvm.script.ir_module
# class MyBmmRelu:
#   @T.prim_func
#   def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"],
#                B: T.Buffer[(16, 128, 128), "float32"],
#                C: T.Buffer[(16, 128, 128), "float32"]):
#     T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
#     Y = T.alloc_buffer((16, 128, 128), dtype="float32")
#     # for n, i, j, k in T.grid(16, 128, 128, 128):
#     for b, i, j, k in T.grid(16, 128, 128, 128):
#       with T.block("Y"):
#         # vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
#         vb, vi, vj, vk = T.axis.remap("SSSR", [b, i, j, k])
#         with T.init():
#           Y[vb, vi, vj] = T.float32(0)
#         Y[vb, vi, vj] += A[vb, vi, vk] * B[vb, vk, vj]

#       with T.block("C"):
#         vb, vi, vj = T.axis.remap("SSS", [b, i, j])
#         C[vb, vi, vj] = T.max(C[vb, vi, vj], T.float32(0))

@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"],
               B: T.Buffer[(16, 128, 128), "float32"],
               C: T.Buffer[(16, 128, 128), "float32"]):
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    Y = T.alloc_buffer((16, 128, 128), dtype="float32")
    # for n, i, j, k in T.grid(16, 128, 128, 128):
    for i0, i1, i2, i3 in T.grid(16, 128, 128, 128):
      with T.block("Y"):
        # vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
        n, i, j, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
        with T.init():
          Y[n, i, j] = T.float32(0)
        Y[n, i, j] += A[n, i, k] * B[n, k, j]

    for i0, i1, i2 in T.grid(16, 128, 128):
      with T.block("C"):
        n, i, j = T.axis.remap("SSS", [i0, i1, i2])
        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))
  
sch = tvm.tir.Schedule(MyBmmRelu)
print(sch.mod.script())

# target tensorIR
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"], B: T.Buffer[(16, 128, 128), "float32"], C: T.Buffer[(16, 128, 128), "float32"]) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))

sch = tvm.tir.Schedule(MyBmmRelu)
# TODO: transformations
# Hints: you can use
# `IPython.display.Code(sch.mod.script(), language="python")`
# or `print(sch.mod.script())`
# to show the current program at any time during the transformation.

# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")

# Step 2. Get loops
b, i, j, k = sch.get_loops(Y)

# Step 3. Organize the loops
k0, k1 = sch.split(k, [32, 4])
j0, j1 = sch.split(j, [16, 8])
sch.reorder(j0, k0, k1, j1)
print(sch.mod.script())

# sch.compute_at/reverse_compute_at(...)
C = sch.get_block("C", func_name="bmm_relu")
sch.reverse_compute_at(C, j0)
print(sch.mod.script())

# Step 4. decompose reduction
sch.parallel(b) # decompose_reduction will break block, so ahead run it
Y_init = sch.decompose_reduction(Y, k0)

print(sch.mod.script())

# Step 5. vectorize / parallel / unroll
Yn, Yb, Yj0, Yj1 = sch.get_loops(Y_init)
_, _, _, Cj1, = sch.get_loops(C)
sch.vectorize(Yj1)
sch.vectorize(Cj1)
# sch.parallel(b) # must before decompose_reduction 
sch.unroll(k1)
print(sch.mod.script())

tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass: transform equal TargetModule")


# eval performance
before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))