import random
from typing import List, Tuple

POP_SIZE: int = 500
MUT_RATE: float = 0.1
TARGET: str = '217y1a05c0'
GENES: str = 'abcdefghijklmnopqrstuvwxyz0123456789'

def initialize_pop() -> List[List[str]]:
    return [[random.choice(GENES) for _ in range(len(TARGET))] for _ in range(POP_SIZE)]

def crossover(selected: List[Tuple[List[str], int]], population: List[Tuple[List[str], int]]) -> List[List[str]]:
    return [random.choice(selected)[0][:point] + random.choice(population[:POP_SIZE//2])[0][point:]
            for point in [random.randint(1, len(TARGET)-1)] for _ in range(POP_SIZE)]

def mutate(offspring: List[List[str]]) -> List[List[str]]:
    return [[random.choice(GENES) if random.random() < MUT_RATE else gene for gene in child] for child in offspring]

def selection(population: List[Tuple[List[str], int]]) -> List[Tuple[List[str], int]]:
    return sorted(population, key=lambda x: x[1])[:POP_SIZE//2]

def fitness_cal(chromo: List[str]) -> Tuple[List[str], int]:
    return chromo, sum(t != c for t, c in zip(TARGET, chromo))

def replace(new_gen: List[Tuple[List[str], int]], population: List[Tuple[List[str], int]]) -> List[Tuple[List[str], int]]:
    return [new if new[1] < old[1] else old for new, old in zip(new_gen, population)]

def main() -> None:
    population: List[Tuple[List[str], int]] = [fitness_cal(chromo) for chromo in initialize_pop()]
    generation: int = 1
    while True:
        selected: List[Tuple[List[str], int]] = selection(population)
        crossovered: List[List[str]] = crossover(selected, population)
        mutated: List[List[str]] = mutate(crossovered)
        new_gen: List[Tuple[List[str], int]] = [fitness_cal(chromo) for chromo in mutated]
        population = replace(new_gen, population)
        best: Tuple[List[str], int] = min(population, key=lambda x: x[1])
        print(f'String: {"".join(best[0])} Generation: {generation} Fitness: {best[1]}')
        if best[1] == 0:
            print('Target found')
            break
        generation += 1

main()
