import numpy as np
import time as time
from ga import GeneticAlgorithm
from ga import fitness_function

def random_knapsack_problem(n=100, scale=10**5, seed=None):
    if seed is not None:
        np.random.seed(seed)    
    items = np.ceil(scale * np.random.rand(n, 2)).astype("int32")
    v = items[:, 0]
    c = items[:, 1]
    C = int(np.ceil(0.5 * 0.5 * n * scale))
    return v, c, C

def knapsack_problem_dp_solve(v, c, C):
    n = v.size
    a = np.zeros((C + 1, n), dtype="int32") # a[i, j] = best pack of knapsack with capacity i using objects from set {0, ..., j - 1}
    b = np.empty((C + 1, n), dtype="object") # back pointers
    for j in range(n):
        b[0, j] = (0, 0)        
    for i in range(1, C + 1):
        if c[0] <= i:
            a[i, 0] = v[0]
            b[i, 0] = (int(i - c[0]), 0)
        else:
            b[i, 0] = (i, -1)
        for j in range(1, n):
            i_prev = int(i - c[j])
            if c[j] > i:
                a[i, j] = a[i, j - 1]
                b[i, j] = (i, j - 1)
            elif a[i, j - 1] >= a[i_prev, j - 1] + v[j]:
                a[i, j] = a[i, j - 1]
                b[i, j] = (i, j - 1)
            else:
                a[i, j] = a[i_prev, j - 1] + v[j]
                b[i, j] = (i_prev, j - 1)    
    solution = np.zeros(n, dtype="int8")
    i = C
    j = n - 1    
    while i > 0 and j >= 0:
        if b[i, j][0] < i:
            solution[j] = 1
        i, j = b[i, j]
    best_pack_value = a[C, n - 1] 
    return best_pack_value, solution                        
            

def prev_main():
    n = 100
    scale = 2000
    history = True
    seed_problem = 0
    v, c, C = random_knapsack_problem(n=n, scale=scale, seed=seed_problem)
    print("RANDOM KNAPSACK PROBLEM:")
    print(f"v: {v}")
    print(f"c: {c}")
    print(f"C: {C}")
    
    print("SOLVING VIA DYNAMIC PROGRAMMING...")
    t1 = time.time()
    best_pack_value, solution = knapsack_problem_dp_solve(v, c, C)
    t2 = time.time()
    print("SOLVING VIA DYNAMIC PROGRAMMING DONE IN: " + str(t2 - t1) + " s.")
    print("BEST PACK VALUE: " + str(best_pack_value))
    print("SOLUTION: " + str(solution))
    print("PACK VALUE CHECK: " + str(solution.dot(v)))
    print("PACK CAPACITY CHECK: " + str(solution.dot(c)))
    
if __name__ == '__main__':
    # Przyk³ad z rozwi¹zaniem plecaka
    n = 100
    scale = 2000
    v, c, C = random_knapsack_problem(n=n, scale=scale)

    print("Obliczanie rozwiazania dokladnego...")
    # Obliczenia dla dok³adnego rozwi¹zania
    best_pack_value, exact_solution = knapsack_problem_dp_solve(v, c, C)
    print(f"Wartosc rozwiazania dokladnego: {best_pack_value}")
    # Uruchomienie algorytmu genetycznego
    #ga = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, generations=100, selection_type="roulette", crossover_type="single", penalty_type="soft")
    #ga = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate=0.5, generations=100, selection_type="roulette", crossover_type="double", penalty_type="soft")
    #ga = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, generations=100, selection_type="rank", crossover_type="single", penalty_type="soft")
    #ga = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate = 0.1, generations=100, selection_type="rank", crossover_type="double", penalty_type="hard")
    ga = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate = 0.01, generations=100, selection_type="eugenic", crossover_type="double", penalty_type="hard")
    best_solution, best_fitness = ga.run(v, c, C, best_pack_value, exact_solution)


    print(f"Najlepsze rozwiazanie genetyczne: {best_solution}")

    # Porównanie rozwi¹zania genetycznego i dok³adnego
    genetic_capacity = np.dot(best_solution, c)
    exact_value = best_pack_value
    exact_capacity = np.dot(exact_solution, c)

    print(f"Wartosc rozwiazania genetycznego: {best_fitness}")
    print(f"Wartosc rozwiazania dokladnego: {exact_value}")
    print(f"Stosunek wartosci genetycznego do dokladnego: {best_fitness / exact_value:.2f}")
    print(f"Procentowa zgodnosc bitowa: {np.sum(best_solution == exact_solution) / n * 100:.2f}%")

    # Rysowanie wykresu przystosowania
    ga.plot_fitness()

    
