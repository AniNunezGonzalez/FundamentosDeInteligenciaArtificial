#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

import torch

#============================================
# Calcular el gradiente de forma automática
#============================================

#===================
# Regresión lineal
# f = w * x 
#===================

#======================
# Ejemplo : f = 2 * x
#======================
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#=========
# Modelo
#=========
def forward(x):
    return w * x

#===================
# Error: loss = MSE
#===================
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

#==============
# Aprendizaje
#==============
learning_rate = 0.01
n_iters = 100

#===========================
for epoch in range(n_iters):
    # Evaluar función
    y_pred = forward(X)
    # Cálcular error
    l = loss(Y, y_pred)
    # Cálcular gradiente
    l.backward()
    # Mejorar coeficientes
    with torch.no_grad():
        w -= learning_rate * w.grad
    # Resetear gradiente
    w.grad.zero_()
    # Diagnóstico
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')

#=======================

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')