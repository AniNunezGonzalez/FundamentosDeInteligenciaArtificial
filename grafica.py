#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

#==========================
# Gráficas de aprendizaje
#==========================
import matplotlib.pyplot as plt
from IPython import display

#plt.ion()

#================================
# Gráfica dinámica en el tiempo
#================================
def graficar(scores, mean_scores):
    display.clear_outpu(wait=True)
    display.display(plt.gcf())
    plt.ion() # Modo interactivo
    plt.clf()
    plt.title('Entrenando...')
    plt.xlabel('Número de Juegos')
    plt.ylabel('Puntaje')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)