#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#==========================================
# Tensor a optimizar -> requires_grad=True
#==========================================
w = torch.tensor(1.0, requires_grad=True)

#==============================
# Evaluación cálculo de costo
#==============================
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

#==========================================
# Retropropagación para calcular gradiente
#==========================================
loss.backward()
print(w.grad)

#========================================
# Nuevos coeficientes
# repetir evaluación y retropropagación
#========================================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()