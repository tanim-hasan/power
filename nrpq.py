#newton raphson 1
import torch

Y = torch.tensor([
    [20-50j, -10+20j, -10+30j],
    [-10+20j, 26-52j, -16+32j],
    [-10+30j, -16+32j, 26-62j]
], dtype=torch.complex64)

S = torch.tensor([0, -256.6 - 110.2j, -138.6 - 45.2j], dtype=torch.complex64) / 100
V_mag = torch.tensor([1.05, 1.0, 1.0])
delta = torch.zeros(3)

for _ in range(20):
    V = V_mag * torch.exp(1j * delta)
    S_calc = V * (Y @ V).conj()
    mis = torch.cat([(S[1:].real - S_calc[1:].real), (S[1:].imag - S_calc[1:].imag)])
    if torch.max(torch.abs(mis)) < 1e-4: break

    J = torch.zeros(4, 4)
    for i in range(1, 3):
        for k in range(1, 3):
            Vi, Vk, Yik = V[i], V[k], Y[i,k]
            d = delta[i] - delta[k]
            if i == k:
                Pi, Qi = S_calc[i].real, S_calc[i].imag
                J[i-1, k-1] = -Qi - V_mag[i]**2 * Y[i,i].imag
                J[i-1, k+1] = Pi / V_mag[i] + V_mag[i] * Y[i,i].real
                J[i+1, k-1] = Pi - V_mag[i]**2 * Y[i,i].real
                J[i+1, k+1] = Qi / V_mag[i] - V_mag[i] * Y[i,i].imag
            else:
                J[i-1, k-1] = V_mag[i]*V_mag[k]*(Yik.real*torch.sin(d) - Yik.imag*torch.cos(d))
                J[i-1, k+1] = V_mag[i]*(Yik.real*torch.cos(d) + Yik.imag*torch.sin(d))
                J[i+1, k-1] = -V_mag[i]*V_mag[k]*(Yik.real*torch.cos(d) + Yik.imag*torch.sin(d))
                J[i+1, k+1] = V_mag[i]*(Yik.real*torch.sin(d) - Yik.imag*torch.cos(d))

    dx = torch.linalg.solve(J, mis)
    delta[1:] += dx[:2]
    V_mag[1:] += dx[2:]

V = V_mag * torch.exp(1j * delta)
for i, v in enumerate(V):
    print(f"V{i+1} = {abs(v):.4f} ∠ {torch.rad2deg(torch.angle(v)).item():.2f}°")
