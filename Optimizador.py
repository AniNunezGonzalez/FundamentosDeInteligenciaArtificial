#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

#==============================================================
# 1) Diseñar el modelo (entrada, salida, NN con muchas capas)
# 2) Definir error y optimizador
# 3) Ciclos de aprendizaje
#       - Forward = evaluar, predecir y calcular error
#       - Backward = calcular gradiente
#       - Mejorar coeficientes
#==============================================================

import torch
import torch.nn as nn

#===================
# Regresión lineal
# f = w * x 
#===================

#=====================
# Ejemplo: f = 2 * x
#=====================

#============================
# 0) Datos de entrenamiento
#============================
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

#=========================================
# 1) Diseño de modelo: coeficientes y NN
#=========================================
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

#=================================
# 2) Definir error y optimizador
#=================================
learning_rate = 0.01
n_iters = 100
# error (loss) definido en pytorch
loss = nn.MSELoss()
# optimizador (SDG stochastic gradient descent)
optimizer = torch.optim.SGD([w], lr=learning_rate)

#==========================
# 3) Ciclo de aprendizaje
#==========================
for epoch in range(n_iters):
    # predict = evaluar función
    y_predicted = forward(X)
    # Error
    l = loss(Y, y_predicted)
    # Calcular gradiente = retropropagación
    l.backward()
    # Mejorar coeficientes
    optimizer.step()
    # Resetear gradiente
    optimizer.zero_grad()
    # Diagnóstico
    if epoch % 10 == 0:
        print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')