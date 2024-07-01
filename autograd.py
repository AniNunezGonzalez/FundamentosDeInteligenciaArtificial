#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

import torch
#=================================================
# Autograd: diferenciación automática de Tensores
#==================================================

#=============================================================
# Bandera requires_grad = True -> genera funciones gradiente
# para las operaciones que se hacen con ese tensor
#=============================================================
x = torch.randn(3, requires_grad=True)
y = x + 2

#====================================
# y = y(x) tiene un grad_fn asociado
#====================================
print(x)
print(y)
print(y.grad_fn)

#==================================
# z = z(y) = z(y(x))
#==================================
z = y * y * 3
print(z)
z = z.mean()
print(z)

#=============================================
# Cálculo del gradiente con retropropagación
#=============================================
z.backward()
print(x.grad)

#===============================================
# Torch.autograd se basa en regla de la cadena
#===============================================
x = torch.randn(3, requires_grad=True)
y = x * 2
for _ in range(10):
    y = y * 2
print(y)
print(y.shape)

#=================================
# evaluar "gradiente" dy/dx en v
#=================================
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

#====================================================
# Decirle a un tensor que deje de generar gradientes
#====================================================
# - x.requires_grad_(False)
# - x.detach()
# - envolverlo con 'with torch.no_grad():'
# .requires_grad_(...) cambia la bandera
#====================================================
a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)
#================
# Con gradiente
#================
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
#===============
# Sin gradiente
#===============
b = a.detach()
print(b.requires_grad)
#================
# Con envoltura
#================
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

#=================================================
# backward() acumula el gradiente en .grad
# .zero_() limpia el gradiente antes de comenzar
#=================================================
weights = torch.ones(4, requires_grad=True)
#==============================
# Epoch: paso de optimización
#==============================
for epoch in range(3):
    # ejemplito
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    # optimización: encontrar nuevos coeficientes
    with torch.no_grad():
        weights -= 0.1 * weights.grad
    # reinicialoiza el gradiente a cero (importante)
    weights.grad.zero_()
print(weights)
print(model_output)

#=================================================
# Optimizer tiene método zero_grad()
# optimizer = torch.optim.SDG([weights], lr=0.1)
# SGD = Stochastic Gradient Descent
# During training:
# optimizer.step()
# optimizer.zero_grad()
#=================================================