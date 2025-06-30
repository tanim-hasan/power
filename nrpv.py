#Newton Raphson 2
import torch

Y = torch.tensor([
    [20-50j, -10+20j, -10+30j],
    [-10+20j, 26-52j, -16+32j],
    [-10+30j, -16+32j, 26-62j]
], dtype=torch.complex64)

S2 = (-400 - 250j) / 100
P3, V3_mag = 2.0, 1.04
V_mag = torch.tensor([1.05, 1.0, V3_mag])
delta = torch.zeros(3)

for _ in range(20):
    V = V_mag * torch.exp(1j * delta)
    S_calc = V * (Y @ V).conj()

    dP2 = S2.real - S_calc[1].real
    dP3 = P3 - S_calc[2].real
    dQ2 = S2.imag - S_calc[1].imag
    mis = torch.tensor([dP2, dP3, dQ2])

    if torch.max(torch.abs(mis)) < 1e-4: break

    d21 = delta[1] - delta[2]
    J = torch.zeros((3, 3))
    J[0,0] = -S_calc[1].imag - V_mag[1]**2 * Y[1,1].imag
    J[0,1] = V_mag[1]*V_mag[2]*(Y[1,2].real*torch.sin(d21) - Y[1,2].imag*torch.cos(d21))
    J[0,2] = S_calc[1].real / V_mag[1] + V_mag[1]*Y[1,1].real
    J[1,0] = V_mag[1]*V_mag[2]*(Y[2,1].real*torch.sin(-d21) - Y[2,1].imag*torch.cos(-d21))
    J[1,1] = -S_calc[2].imag - V_mag[2]**2 * Y[2,2].imag
    J[1,2] = 0
    J[2,0] = S_calc[1].real - V_mag[1]**2 * Y[1,1].real
    J[2,1] = -V_mag[1]*V_mag[2]*(Y[1,2].real*torch.cos(d21) + Y[1,2].imag*torch.sin(d21))
    J[2,2] = S_calc[1].imag / V_mag[1] - V_mag[1]*Y[1,1].imag

    dx = torch.linalg.solve(J, mis)
    delta[1] += dx[0]
    delta[2] += dx[1]
    V_mag[1] += dx[2]

V = V_mag * torch.exp(1j * delta)
for i, v in enumerate(V):
    print(f"V{i+1} = {abs(v):.4f} ∠ {torch.rad2deg(torch.angle(v)).item():.2f}°")
