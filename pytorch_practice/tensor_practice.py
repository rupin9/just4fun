
import torch
import numpy as np

shape = (2,3,)
torch.manual_seed(2022)
rand_tensor = torch.rand(shape, dtype=torch.float16)
ones_tensor = torch.ones(shape, dtype=torch.float16)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")

r = torch.rand(3,3, dtype=torch.float32)
print(f"r: \n {r} \n")
print("\nAbsolute values of r:")
print(torch.abs(r))
print("\nDeterminant of r:")
print(torch.det(r))
print("\nSingular values of r:")
print(torch.svd(r))
print("\nStandard deviation and average of r:")
print(torch.std_mean(r))

print("\n")

#The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
ones_tensor_fp32 = torch.ones(shape)
ones_tensor_fp16 = torch.ones_like(ones_tensor)

print(f"Ones fp32 Tensor: \n {ones_tensor_fp32}")
print(f"Ones fp16 Tensor: \n {ones_tensor_fp16}")

print(ones_tensor_fp32.dtype)
print(ones_tensor_fp16.dtype)
print('\n')
print(f"Shape of rand_tensor: {rand_tensor.shape}")
print(f"Datatype of rand_tensor: {rand_tensor.dtype}")
print(f"Device rand_tensor is stored on: {rand_tensor.device}")


tensor = torch.ones(3, 4)
tensor[1,2]=3;

y1 = tensor.T
print(tensor)
print(y1)

# Tensor to NumPy array (same memory)
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # tensor -> numpy
print(f"n: {n}")

t.add_(2) # in-place
print(f"updated n: {n}")

# NumPy to Tensor array (same memory)
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")



