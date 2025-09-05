import numpy as np
import matplotlib.pyplot as plt

# Pontos dados
x_vals = np.array([1, 2])
y_vals = np.array([3, 5])

# Intervalos de m e b
m_vals = np.arange(-10, 10.1, 0.1)
b_vals = np.arange(-10, 10.1, 0.1)

# Preparar grid
M, B = np.meshgrid(m_vals, b_vals)
RMSE = np.zeros_like(M)

# Calcular RMSE para cada combinação de m e b
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        m = M[i, j]
        b = B[i, j]
        y_pred = m * x_vals + b
        rmse = np.sqrt(np.mean((y_vals - y_pred) ** 2))
        RMSE[i, j] = rmse

# Plotar gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, B, RMSE, cmap='viridis')

ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('RMSE')
ax.set_title('RMSE em função de m e b')

plt.show()
