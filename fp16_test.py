import torch
import numpy as np
import struct

print('fp32 => int32, fp16 => int16 =============================\n')
# refert to https://docs.python.org/3/tutorial/floatingpoint.html
#           https://docs.python.org/3/library/stdtypes.html#type-objects
#           https://docs.python.org/3/library/struct.html#module-struct
## fp32 <-> fp16 numpy <--- not easy in python
## b'\x00\x80\xfc?\x00\xa01A' ??? what this is... fp32 0x3ffc8000 0x4131a000
np.set_printoptions(precision=10)
p0 = np.float32([[1.97265625, -11.1015625]]); print('1st float32 p0: ', p0, p0.tobytes(), '\n')
p1 = np.float16(p0); print('2nd float32 -> float16 p1: ', p1, p1.tobytes(), '\n')
p2 = np.float32(p1); print('3rd float16 -> float32 p2: ', p2, p2.tobytes(), '\n')

q0 = struct.unpack('<'+str(p0.size)+'I', p0.tobytes()); print('q0 tuple (dec) =', q0);
q1 = struct.unpack('<'+str(p1.size)+'H', p1.tobytes()); print('q1 tuple (dec) =', q1);

r0 = np.uint32(q0).reshape(p0.shape); print('r0 (dec) =', r0)
r1 = np.uint16(q1).reshape(p0.shape); print('r1 (dec) =', r1)
np.set_printoptions(formatter={'int':hex})
print('r0 (hex) =', r0)
print('r1 (hex) =', r1)
np.set_printoptions(formatter={'int':None})

print('\n\nint32 <=> fp32, int16 <=> fp16 =============================\n')
print(type(r0), 'r0 = ', r0)
print(type(r1), 'r1 = ', r1)

s0 = np.frombuffer(r0.tobytes(), dtype=np.float32).reshape(p0.shape)
s1 = np.frombuffer(r1.tobytes(), dtype=np.float16).reshape(p0.shape)
print(type(s0), 's0 = ', s0)
print(type(s1), 's1 = ', s1)

t0 = np.frombuffer(s0.tobytes(), dtype=np.int32).reshape(p0.shape)
t1 = np.frombuffer(s1.tobytes(), dtype=np.int16).reshape(p0.shape)
print(type(t0), 't0 = ', t0)
print(type(t1), 't1 = ', t1)

u0 = np.frombuffer(s0.tobytes(), dtype=np.int32)
u1 = np.frombuffer(s1.tobytes(), dtype=np.int16)
print(type(u0), 'u0 = ', u0)
print(type(u1), 'u1 = ', u1)

##########################################################
## 3D tensor
p0 = torch.tensor([[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]], [[0.1, -0.023], [1.8, -91.1]]], dtype=torch.float16)
print('3D tensor(float16): ', p0, '\n')

p1 = np.float16(p0)
p2 = struct.unpack('<'+str(p1.size)+'H', p1.tobytes())
p3 = torch.tensor(np.float16(p2).reshape(p0.shape), dtype=torch.int16)
print('3D tensor(int16): ', p3, '\n')

np.set_printoptions(formatter={'int':hex})
print('np.array(tensor) hex format print =', np.array(p3).reshape(p0.shape), '\n')
np.set_printoptions(formatter={'int':None})
print('np.array(tensor) dec format print =', np.array(p3).reshape(p0.shape), '\n')

## Here
t1 = np.frombuffer(p1.tobytes(), dtype=np.uint16).reshape(p1.shape) ## ndarray uint16
#a = torch.tensor(t1, dtype=torch.int16) ## error
#a = torch.tensor(t1) ## error
s1 = np.frombuffer(t1.tobytes(), dtype=np.float16).reshape(p1.shape)
q0 = torch.tensor(s1)
print('t1 = ', t1)
print(s1)
print(p1-s1)
print(p0-q0)