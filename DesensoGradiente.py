#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

import numpy as np 

#=======================
# Calcular manualmente
#=======================

#===================
# Regresión lineal
# f = w * x
#===================

#=====================
# ejemplo: f = 2 * x
#=====================
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

#=========
# Modelo
#=========
def forward(x):
    return w * x

#====================
# Error: loss = MSE
#====================
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

#===============================
# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
#===============================
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#==============
# Aprendizaje
#==============
learning_rate = 0.01 # Coeficiente de aprendizaje
n_iters = 20         # Iteraciones

#===========================
for epoch in range(n_iters):
    # pedicción = evaluar función
    y_pred = forward(X)
    # Cálculo de error
    l = loss(Y, y_pred)
    # Cálcular gradiente
    dw = gradient(X, Y, y_pred)
    # Mejorar coeficientes
    w -= learning_rate * dw
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')