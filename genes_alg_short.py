import random

POP_SIZE = 500
MUT_RATE = 0.1
TARGET = '217y1a05c0'
GENES = ' abcdefghijklmnopqrstuvwxyz0123456789'

def initialize_pop():
    return [[''.join(random.choice(GENES) for _ in range(len(TARGET)))] for _ in range(POP_SIZE)]

def crossover(selected, population):
    return [random.choice(selected)[0][:point] + random.choice(population[:POP_SIZE//2])[0][point:]
            for point in [random.randint(1, len(TARGET)-1)] for _ in range(POP_SIZE)]

def mutate(offspring):
    return [[''.join(random.choice(GENES) if random.random() < MUT_RATE else gene for gene in child)] for child in offspring]

def selection(population):
    return sorted(population, key=lambda x: x[1])[:POP_SIZE//2]

def fitness_cal(chromo):
    return [chromo, sum(t != c for t, c in zip(TARGET, chromo))]

def replace(new_gen, population):
    return [new if new[1] < old[1] else old for new, old in zip(new_gen, population)]

def main():
    population = [fitness_cal(chromo) for chromo in initialize_pop()]
    generation = 1
    while True:
        selected = selection(population)
        crossovered = crossover(selected, population)
        mutated = mutate(crossovered)
        new_gen = [fitness_cal(chromo) for chromo in mutated]
        population = replace(new_gen, population)
        best = min(population, key=lambda x: x[1])
        print(f'String: {best[0]} Generation: {generation} Fitness: {best[1]}')
        if best[1] == 0:
            print('Target found')
            break
        generation += 1

main()
