## Rede OR com GPU e parada pelo erro
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

input_samples = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)

output_target = np.array([0, 1, 1, 1])

class Perceptron():
    def __init__(self, inputs, desired_outputs, learning_rate=0.025, error_threshold=1.0e-3):
        self.weights = np.random.rand(3)  # Pesos iniciais (2 entradas + bias)
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.desired_outputs = desired_outputs
        self.error_threshold = error_threshold

        # Configuração do gráfico
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.grid(True)

        # Pontos de entrada
        for input, desired_output in zip(self.inputs, self.desired_outputs):
            self.ax.plot(input[0], input[1], 'ro' if desired_output == 1 else 'go', markersize=10)

        # Linha de decisão inicial
        self.line, = self.ax.plot([], [], 'b-', label='Linha de decisão')

    def update_decision_boundary(self):
        """Atualiza a linha de decisão no gráfico."""
        x = np.linspace(-3, 3, 10)
        if self.weights[1] != 0:
            y = -(self.weights[0] * x + self.weights[2]) / self.weights[1]
        else:
            y = np.zeros_like(x)
        self.line.set_data(x, y)  # Converta para CPU para plotar

    def activation_func(self, x):
        if x >= 0:
            return 1
        elif x < 0:
            return 0


    def train(self):
        total_error = np.inf
        epoch = 0

        def update(frame):
            nonlocal total_error, epoch
            total_error = 0

            for input, output in zip(self.inputs, self.desired_outputs):
                input = np.append(input, 1)  # Adiciona o bias na entrada

                y = self.activation_func(self.weights @ input)
                error = output - y
                self.weights += self.learning_rate * error * input
                total_error += np.abs(error)

            epoch += 1
            print(f"Epoch: {epoch}, Total Error: {total_error:.4f}")

            self.update_decision_boundary()
            self.ax.set_title(f'Epoch: {epoch}, Total Error: {total_error:.4f}')

            if total_error < self.error_threshold:
                print("Treinamento concluído.")
                #plt.close(self.fig)  # Fecha o gráfico ao atingir o limite
            return self.line,

        # Animação com limite de epochs para evitar loop infinito
        ani = FuncAnimation(self.fig, update, frames=range(500), blit=False, repeat=False)
        plt.legend()
        plt.show()

    def predict(self, input):
        input = np.append(input, 1)  # Adiciona o bias
        y = self.weights @ input
        return int(y >= 0.5)

p = Perceptron(input_samples, output_target)
p.train()

# Teste o modelo
print(f"Pesos finais: {p.weights}")
print(
    [
        p.predict(np.array([0, 0])), 
        p.predict(np.array([0, 1])), 
        p.predict(np.array([1, 0])), 
        p.predict(np.array([1, 1]))
    ]
)
