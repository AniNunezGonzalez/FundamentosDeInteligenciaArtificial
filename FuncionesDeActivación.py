#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

#============================================================
# Introducción al uso de funciones de activación en pytorch
#============================================================
# output = w*x + b
# output = activation_function(output)
#============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

#=================
# uso de softmax
#=================
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

#===========
# sigmoide
#===========
output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

#=====
#tanh
#=====
output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)

#======
# relu
#======
output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)

#=============
# leaky relu
#=============
output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)

#=======================================================================================
# nn.ReLU() crea un nn.Module que puede añadir (por ejemplo) a un modelo nn.Sequential
# torch.relu solo es la llamada abstracta a la función relu
# para que lo añadas (por ejemplo) a tu evañuación (forward method)
#=======================================================================================
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

#=======================================================================
# opción 2: usar funciones de activación directamente en la evaluación
#=======================================================================
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out