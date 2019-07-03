import random as rnd
import math
from Individual import Individual
import numpy as np
import operator
import matplotlib.pyplot as plt

ij = 0.1
input = []
for i in range(-100,100):
    input.append(ij + i*0.1)
## Y = 25x^3 + 3x^2 + 2x + 2
output = [(25*(math.pow(item,3)) + 3*(math.pow(item,2)) + 2*item + 2) for item in input]
populationSize = 50
numberOfGenerations = 500
tornumentSize = 8
mutationRate = [0.01, 0.02, 0.05, 0.1]
sigma = [10 ** -3, 10 ** -2, 10 ** -1, 1, 10]
# selected options
sigma_selected = sigma[2]
mutationRate_selected = 0.8


def MSE(prediction):
    return np.square(np.array(prediction) - np.array(output)).mean(axis=0)


def fitness(individual):
    prediction = []
    for i in range(len(output)):
        prediction.append(individual.calculate(input[i]))
    return 1 + MSE(prediction)


def generate_crossover(individuals):
    new_pop = []
    for i in range(len(individuals) - 1):
        new_pop.append(Individual(individuals[i].a0, individuals[i + 1].a1, individuals[i].a2, individuals[i + 1].a3))
    return new_pop


def mutation(individuals):
    for i in range(len(individuals)):
        individuals[i].a0 += np.random.normal(0, sigma_selected, 1)[0] if rnd.randint(0,
                                                                                      100) / 100 > mutationRate_selected else 0
        individuals[i].a1 += np.random.normal(0, sigma_selected, 1)[0] if rnd.randint(0,
                                                                                      100) / 100 > mutationRate_selected else 0
        individuals[i].a2 += np.random.normal(0, sigma_selected, 1)[0] if rnd.randint(0,
                                                                                      100) / 100 > mutationRate_selected else 0
        individuals[i].a3 += np.random.normal(0, sigma_selected, 1)[0] if rnd.randint(0,
                                                                                      100) / 100 > mutationRate_selected else 0
        individuals[i].fitness = fitness(individuals[i])


generation = []
for i in range(populationSize):
    generation.append(Individual(rnd.randint(-10, 10), rnd.randint(-10, 10), rnd.randint(0, 10), rnd.randint(0, 10)))
report = []
for numberOfRemainingGeneration in range(numberOfGenerations):
    for individual in generation:
        individual.fitness = fitness(individual)
    generation_sorted = sorted(generation, key=operator.attrgetter('fitness'))
    population_selected = generation_sorted[0:tornumentSize]
    report.append({'generation': numberOfRemainingGeneration,
                   'mean_fitness': np.mean(np.array([item.fitness for item in generation])),
                   'worst_fitness': generation_sorted[len(generation_sorted) - 1].fitness,
                   'best_fitness': generation_sorted[0].fitness})
    new_generation = generate_crossover(population_selected)
    mutation(new_generation)
    all_gens = new_generation + generation
    all_gens_sorted = sorted(all_gens, key=operator.attrgetter('fitness'))
    generation = all_gens_sorted[0:populationSize]
mean_MSE = [item['mean_fitness'] for item in report]
worst_MSE = [item['worst_fitness'] for item in report]
best_MSE = [item['best_fitness'] for item in report]

prediction = []
for i in range(len(output)):
    prediction.append(generation[0].calculate(input[i]))

fig,axs = plt.subplots(1, constrained_layout=True)
axs.plot(input, output, '--', input, prediction, 'o')
axs.set_title('prediction')
axs.set_xlabel('x')
axs.set_ylabel('Y,Yp')
plt.show()
fig,axs = plt.subplots(1, constrained_layout=True)
axs.plot(mean_MSE, '--')
axs.set_xlabel('time ')
axs.set_title('mean fitness')
axs.set_ylabel('fitness')
plt.show()
fig,axs = plt.subplots(1, constrained_layout=True)
axs.plot(worst_MSE, '--')
axs.set_xlabel('time ')
axs.set_title('worst fitness')
axs.set_ylabel('fitness')
plt.show()
fig,axs = plt.subplots(1, constrained_layout=True)
axs.plot(best_MSE, '--')
axs.set_xlabel('time ')
axs.set_title('best fitness')
axs.set_ylabel('fitness')
plt.show()
