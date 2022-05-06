
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

grad_en = 1

if grad_en:
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w)+b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    loss.backward()
    print(f"\nx={x}")
    print(f"\nw={w}")
    print(f"\nb={b}")
    print(f"\nw.grad={w.grad}")
    print(f"\nb.grad={b.grad}")
else:
    with torch.no_grad():
        w = torch.randn(5, 3)
        b = torch.randn(3)
        z = torch.matmul(x, w)+b
        print(f"\nz.requires_grad = {z.requires_grad} or")
        z_det = z.detach()
        print(f"\nz_det.requires_grad = {z_det.requires_grad}")
        print(f"\nx={x}")
        print(f"\nw={w}")
        print(f"\nb={b}")
   
print(f"\nz={z}")


# Interesting point in pytorch: 
# DAGs are dynamic in PyTorch An important thing to note is that 
# the graph is recreated from scratch; after each .backward() call, 
# autograd starts populating a new graph. This is exactly what allows 
# you to use control flow statements in your model; you can change the shape, 
# size and operations at every iteration if needed.

################################################
# We can only perform gradient calculations using backward once on a given graph, 
# for performance reasons. If we need to do several backward calls on the same graph, 
# we need to pass retain_graph=True to the backward call.

x = torch.eye(5, requires_grad=True) # identity matrix
out = (x+1).pow(2) # (x+1)^2
print(f"out\n{out}")
out.backward(torch.ones_like(x), retain_graph=True) # 2*(x+1)
print(f"First call\n{x.grad}")
out.backward(torch.ones_like(x), retain_graph=True)
print(f"\nSecond call\n{x.grad}") # <= accumulated output
out.backward(torch.ones_like(x), retain_graph=True)
print(f"\nThird call\n{x.grad}") # <= accumulated output


x.grad.zero_() # reset
out.backward(torch.ones_like(x), retain_graph=True)
print(f"\nCall after zeroing gradients\n{x.grad}")


