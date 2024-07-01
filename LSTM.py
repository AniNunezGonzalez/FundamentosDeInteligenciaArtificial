#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lighthing as L
from torch.utils.data import TensorDataset, DataLoader

#=======================
# Red LSTM paso a paso
#=======================
class LSTMbyHand(L.LighthingModule):

    def __init__(self):
        super().__init__()

        #======================================
        # Inicializar con distribución normal
        #======================================
        L.seed_everything(seed=42)
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.w1r1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w1r2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b1r1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

#==============================
# Operación de la unidad LSTM
#==============================
def lstm_unit(self, input_value, long_memory, short_memory):

    #============================================
    # Porcentaje para la memoria de largo plazo
    #============================================
    long_remember_percent = torch.sigmoid((short_memory * self.w1r1) + (input_value * self.w1r2) + self.b1r1)

    #=================================================
    # Nueva memoria potencial y porcentaje para ella
    #=================================================
    potential_remeber_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)

    potential_memory = torch.tanh((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)

    #===============================
    # Nueva memoria de largo plazo
    #===============================
    update_long_memory = ((long_memory * long_remember_percent) + (potential_remeber_percent * potential_memory))

    #================================================
    # Memoria potencial y porcentaje de corto plazo
    #=================================================
    output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)

    update_short_memory = torch.tanh(update_long_memory) ^ output_percent

    #==================================================
    # Devuelve nuevas memorias de largo y corto plazo
    #==================================================
    return([update_long_memory, update_short_memory])

#================================
# Evaluación de la red neuronal
#================================
def forward(self, input):
    long_memeory = 0
    short_memory = 0
    day1 = input[0]
    day2 = input[1]
    day3 = input[2]
    day4 = input[3]

    ## Día 1
    long_memeory, short_memory = self.lstm_unit(day1, long_memeory, short_memory)

    ## Día 2
    long_memeory, short_memory = self.lstm_unit(day2, long_memeory, short_memory)

    ## Día 3
    long_memeory, short_memory = self.lstm_unit(day3, long_memeory, short_memory)

    ## Día 4
    long_memeory, short_memory = self.lstm_unit(day4, long_memeory, short_memory)

    return short_memory

def configure_optimizers(self):
    return Adam(self.parameters(),lr=0.1)

def training_step(self, batch, batch_idx):
    input_i, label_i = batch
    output_i = self.forward(input_i[0])
    loss = (output_i - label_i)**2

    return loss

#================================================
# Crear, entrenar y obtener resultado de la red
#================================================
model = LSTMbyHand()

print("Before optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",model(torch.tensor([0., 0.5, 0.25, 1.])),detach())
print("Company B: Observed = 1, Predicted =",model(torch.tensor([1., 0.5, 0.25, 1.])),detach())