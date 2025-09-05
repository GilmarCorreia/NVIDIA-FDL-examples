import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Pontos conhecidos
x_vals = np.array([1, 2])
y_vals = np.array([3, 5])

# Inicialização da figura
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('RMSE')
ax.set_title('RMSE em função de m e b (acúmulo animado com cores)')

# Lista para acumular os pontos
m_all = []
b_all = []
rmse_all = []

# Número total de pontos desejado
total_points = 44100  # 201x201 = toda a grade de -10 a 10 com passo de 0.1
points_per_frame = 10
frames = total_points // points_per_frame

# Pré-gerar todas as combinações possíveis
m_vals = np.arange(-10, 10.1, 0.1)
b_vals = np.arange(-10, 10.1, 0.1)
m_grid, b_grid = np.meshgrid(m_vals, b_vals)
m_flat = m_grid.ravel()
b_flat = b_grid.ravel()
indices = np.random.permutation(len(m_flat))  # embaralhar os pares

def update(frame):
    start = frame * points_per_frame
    end = start + points_per_frame
    if end > len(indices):
        end = len(indices)

    for idx in indices[start:end]:
        m = m_flat[idx]
        b = b_flat[idx]
        y_pred = m * x_vals + b
        rmse = np.sqrt(np.mean((y_vals - y_pred) ** 2))

        m_all.append(m)
        b_all.append(b)
        rmse_all.append(rmse)

    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 30)
    ax.set_xlabel('m')
    ax.set_ylabel('b')
    ax.set_zlabel('RMSE')
    ax.set_title('RMSE em função de m e b (acúmulo animado com cores)')

    sc = ax.scatter(m_all, b_all, rmse_all, c=rmse_all, cmap='viridis', s=2)

    # Adiciona barra de cores (apenas uma vez)
    # if frame == 0:
    #     fig.colorbar(sc, ax=ax, label='RMSE')

    if end == len(indices):
        ani.event_source.stop()

ani = FuncAnimation(fig, update, frames=frames, interval=1)

plt.show()
