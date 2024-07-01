#=================================================================
# Alumna:Aniela Montserrat Núñez González
# Grupo: 5AV1
# Unidad de aprendizje: Fundamentos de Inteligencia Artifical
# Profesor: Julian Tercero Becerra Sagredo
# 30/06/2024
#=================================================================

import random
import datetime

random.seed(random.random())
startline = datetime.datetime.now()

#============
# Los genes
#============
geneSet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!'

#===========
# Objetivo
#===========
target = "Hola Mundo"

#================
# Frase inicial
#================
def generate_parent(length):
    genes = []
    while len(genes) < length:
        sample_size = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sample_size))
    return ''.join(genes)

#======================
# Función de aptitud
#======================
def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)

#=================================
# Mutación de letras en la frase
#=================================
def mutate(parent):
    index = random.randrange(0, len(parent))
    child_genes = list(parent)
    new_gene, alternate = random.sample(geneSet, 2)
    child_genes[index] = alternate if new_gene == child_genes[index] else new_gene
    return ''.join(child_genes)

#========================
# Monitorea la solución
#========================
def display(guess):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}\t{}\t{}".format(guess,fitness,timeDiff))

#===================
# Código principal
#===================
    bestParent = generate_parent(len(target))
    bestFitness = get_fitness(bestParent)
    display(bestParent)

    #==============
    # Iteraciones
    #==============
    while True:
        child = mutate(bestParent)
        childFitness = get_fitness(child)
        display(child)
        if best_fitness >= childFitness:
            continue
        if childFitness >= len(bestParent):
            break
        bestFitness = childFitness
        bestParent = child