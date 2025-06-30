#kron reduction 2
import torch

# Impedance matrix (z) with infinities
z = torch.tensor([
    [-1.25j, 0, -0.25j, -0.2j],
    [0, -1.25j, -0.4j, -0.2j],
    [-0.25j, -0.4j, float('inf'), -0.125j],
    [-0.2j, -0.2j, -0.125j, float('inf')]
], dtype=torch.complex64)

# Admittance matrix (y) = 1 / z where z ≠ inf and z ≠ 0
y = torch.where(torch.isinf(z), 0, torch.where(z == 0, float('inf'), 1 / z))

# Ybus calculation: diagonal = row sum, off-diagonal = -y
Y = torch.diag(y.sum(dim=1)) - y

print("\nImpedance Matrix (Z):")
print(z)
print("\nAdmittance Matrix (Y):")
print(y)
print("\nBus Admittance Matrix (Ybus):")
print(Y)

# Interactive Kron reduction
while input("\nDo you want to perform reduction? (yes/no): ").lower() == 'yes':
    node = int(input(f"Enter node number to eliminate (1 to {Y.shape[0]}): ")) - 1
    if 0 <= node < Y.shape[0]:
        keep = [i for i in range(Y.shape[0]) if i != node]
        Yred = Y[keep][:, keep] - (Y[keep][:, node:node+1] @ Y[node:node+1, keep]) / Y[node, node]
        Y = Yred
        print("\nReduced Bus Admittance Matrix (Ybus):")
        print(Y)
    else:
        print("Invalid node.")
