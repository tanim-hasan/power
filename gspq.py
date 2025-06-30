#gauss seidal PQ 3
import torch

z = torch.tensor([
    [float('inf'), 0.02 + 0.04j, 0.01 + 0.03j],
    [0.02 + 0.04j, float('inf'), 0.0125 + 0.025j],
    [0.01 + 0.03j, 0.0125 + 0.025j, float('inf')]
], dtype=torch.complex128)

y = torch.where(torch.isinf(z), 0, torch.where(z == 0, float('inf'), 1 / z))
Y = torch.diag(y.sum(dim=1)) - y

v1 = torch.tensor(1.05, dtype=torch.complex128)
v2 = torch.tensor([1], dtype=torch.complex128)
v3 = torch.tensor([1], dtype=torch.complex128)
s2 = -(256.6 + 110.2j) / 100
s3 = -(138.6 + 45.2j) / 100

print('  Iter.                 V2                       V3')
for k in range(20):
    v2_new = ((s2.real - 1j * s2.imag) / v2[-1].conj() - Y[1, 0] * v1 - Y[1, 2] * v3[-1]) / Y[1, 1]
    v3_new = ((s3.real - 1j * s3.imag) / v3[-1].conj() - Y[2, 0] * v1 - Y[2, 1] * v2_new) / Y[2, 2]
    v2 = torch.cat([v2, v2_new.unsqueeze(0)])
    v3 = torch.cat([v3, v3_new.unsqueeze(0)])
    print(f'\t{k+1}\t{v2_new.item():.6f}\t{v3_new.item():.6f}')
    if torch.abs(v2[-1] - v2[-2]) < 1e-5 and torch.abs(v3[-1] - v3[-2]) < 1e-5:
        break

print(f'\nFinal V2: {v2[-1]:.6f} = {abs(v2[-1]):.6f} ∠ {torch.rad2deg(torch.angle(v2[-1])).item():.2f}°')
print(f'Final V3: {v3[-1]:.6f} = {abs(v3[-1]):.6f} ∠ {torch.rad2deg(torch.angle(v3[-1])).item():.2f}°')
