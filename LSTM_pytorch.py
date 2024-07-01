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
from torch.utils.data import Tensordataset, DataLoader

#=========================
# Red LSTM con Lighthing
#=========================
class LighthingLSTM(L.LighthingModule):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    #================================
    # Evaluación de la red neuronal
    #================================
    def forward(self, input):
        input_trans = input.view(len(input), 1)
        lstm_out, temp = self.lstm(input_trans)
        prediction = lstm_out[-1]
        return prediction
    
    #==================================
    # Método de descenso de gradiente
    #==================================
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    #========================
    # Paso de entrenamiento
    #========================
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2
        return loss

#================================================
# Crear, entrenar y obtener resultado de la red
#================================================
model = LighthingLSTM()

print("\nNow let's compare the observed and predicted values...")

print("Company A: Observed = 0, Predicted =",model(torch.tensor([0., 0.5, 0.25, 1.])),detach())
print("Company B: Observed = 1, Predicted =",model(torch.tensor([1., 0.5, 0.25, 1.])),detach())

## create the training data for the neural network.
inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tesor([0., 1.])

dataset = Tensordataset(inputs, labels)
dataloader = torch.tensor([0., 1.])

trainer = L.Trainer(max_epochs=300)
trainer.fit(model, train_dataloaders=dataloader)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",model(torch.tensor([0., 0.5, 0.25, 1.])),detach())
print("Company B: Observed = 1, Predicted =",model(torch.tensor([1., 0.5, 0.25, 1.])),detach())

