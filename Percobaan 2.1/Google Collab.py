import numpy as np
import matplotlib.pyplot as plt
# Mengimpor library numpy dan matplotlib untuk operasi matematik dan plotting grafik

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
# Dataset input dan target output untuk fungsi logika AND.

w = np.random.rand(2)
b = np.random.rand(1)
lr = 0.1
epochs = 20
# Inisialisasi bobot, bias, learning rate, dan epoch training.

def step(x):
    return 1 if x >= 0 else 0
# Fungsi aktivasi step.

error_history = []
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = step(z)
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error
        total_error += abs(error)
    error_history.append(total_error)
    print(f"Epoch {epoch+1} | Total Error: {total_error} | w: {w} | b: {b}")
# Proses training perceptron dan update bobot, bias, serta menghitung total error tiap epoch

np.save('weights.npy', w)
np.save('bias.npy', b)
# Menyimpan bobot dan bias hasil training.

plt.plot(error_history, marker='o')
plt.title("Grafik Total Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.grid(True)
plt.show()
