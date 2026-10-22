import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)  # same result every run


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)


model = PINN()
print(model)

# Collocation points
x = torch.FloatTensor(1000, 1).uniform_(0, 1)
t = torch.FloatTensor(1000, 1).uniform_(0, 1)

# Initial condition: u(x,0) = sin(pi x)
x_ic = torch.FloatTensor(100, 1).uniform_(0, 1)
t_ic = torch.zeros(100, 1)
u_ic = torch.sin(np.pi * x_ic)

# Boundary condition: u(0,t) = 0, u(1,t) = 0
t_bc = torch.FloatTensor(100, 1).uniform_(0, 1)
x_bc0 = torch.zeros(100, 1)
x_bc1 = torch.ones(100, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------
# Training loop
# ---------------------------------
epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    # PDE residual: u_t - u_xx = 0
    f = u_t - u_xx
    loss_pde = torch.mean(f**2)

    # Initial condition loss
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)

    # Boundary condition loss
    u_pred_bc0 = model(x_bc0, t_bc)
    u_pred_bc1 = model(x_bc1, t_bc)
    loss_bc = torch.mean(u_pred_bc0**2) + torch.mean(u_pred_bc1**2)

    loss = loss_pde + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

print("Training Finished!")

# ---------------------------------
# Error vs exact solution: u = exp(-pi^2 t) * sin(pi x)
# ---------------------------------
xs = torch.linspace(0, 1, 101)
ts = torch.linspace(0, 1, 101)
X, T = torch.meshgrid(xs, ts, indexing="ij")
x_g, t_g = X.reshape(-1, 1), T.reshape(-1, 1)

with torch.no_grad():
    u_grid = model(x_g, t_g)
u_true = torch.exp(-np.pi**2 * t_g) * torch.sin(np.pi * x_g)

rel_l2 = torch.norm(u_grid - u_true) / torch.norm(u_true)
max_err = torch.max(torch.abs(u_grid - u_true))
print(f"Relative L2 error: {rel_l2.item():.2e}")
print(f"Max abs error: {max_err.item():.2e}")

# ---------------------------------
# Plot at t = 0.5 (PINN vs exact)
# ---------------------------------
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
t_test = torch.full_like(x_test, 0.5)

with torch.no_grad():
    u_pred = model(x_test, t_test)
u_exact = np.exp(-np.pi**2 * 0.5) * np.sin(np.pi * x_test.numpy())

plt.figure(figsize=(8, 5))
plt.plot(x_test.numpy(), u_pred.numpy(), label="PINN Prediction")
plt.plot(x_test.numpy(), u_exact, "--", label="Exact Solution")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Heat equation solution at t = 0.5")
plt.legend()
plt.grid(True)
plt.savefig("heat_pinn_result.png", dpi=150)
plt.show()