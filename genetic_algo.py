import numpy as np
import time as time
from ga import GeneticAlgorithm
from ga import fitness_function
import matplotlib.pyplot as plt

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
    # Przyk�ad z rozwiazaniem plecaka
    n = 100
    scale = 2000
    v, c, C = random_knapsack_problem(n=n, scale=scale)
    
    print("RANDOM KNAPSACK PROBLEM:")
    print(f"v: {v}")
    print(f"c: {c}")
    print(f"C: {C}")
    
    print("\n\nTestowane algorytmy genetyczne:")
    print("RUSS - Selekcja ruletkowa, krzyzowanie jednopunktowe, lagodna kara")
    print("RUDH - Selekcja ruletkowa, krzyzowanie dwupunktowe, drastyczna kara")
    print("RKSS - Selekcja rankingowa, krzyzowanie jednopunktowe, lagodna kara")
    print("RKDH - Selekcja rankingowa, krzyzowanie dwupunktowe, drastyczna kara")
    #print("XD - Selekcja tylko 2 najlepszych osobnikow, krzyzowanie dwupunktowe, drastyczna kara")
    print("\n\nObliczanie rozwiazania dokladnego poprzez programowanie dynamiczne...", end="\r")
    # Obliczenia dla dok�adnego rozwi�zania
    start_time = time.time()
    best_pack_value, exact_solution = knapsack_problem_dp_solve(v, c, C)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Obliczanie rozwiazania dokladnego poprzez programowanie dynamiczne... {duration:.2f}s")

    # Uruchomienie algorytmu genetycznego
    gaRUSS = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate=0.001, generations=100, selection_type="roulette", crossover_type="single", penalty_type="soft")
    gaRUDH = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate=0.001, generations=100, selection_type="roulette", crossover_type="double", penalty_type="hard")
    gaRKSS = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate=0.001, generations=100, selection_type="rank", crossover_type="single", penalty_type="soft")
    gaRKDH = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate=0.001, generations=100, selection_type="rank", crossover_type="double", penalty_type="hard")
    #gaXD = GeneticAlgorithm(n=n, fitness_function=fitness_function, population_size=1000, mutation_rate=0.01, generations=100, selection_type="eugenic", crossover_type="double", penalty_type="hard")
    print("\nObliczanie RUSS")
    start_time = time.time()
    best_solutionRUSS, best_fitnessRUSS = gaRUSS.run(v, c, C, best_pack_value, exact_solution)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generacja: 100/100 {duration:.2f}s")
    print("\nObliczanie RUDH")
    start_time = time.time()
    best_solutionRUDH, best_fitnessRUDH = gaRUDH.run(v, c, C, best_pack_value, exact_solution)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generacja: 100/100 {duration:.2f}s")
    print("\nObliczanie RKSS")
    start_time = time.time()
    best_solutionRKSS, best_fitnessRKSS = gaRKSS.run(v, c, C, best_pack_value, exact_solution)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generacja: 100/100 {duration:.2f}s")
    print("\nObliczanie RKDH")    
    start_time = time.time()
    best_solutionRKDH, best_fitnessRKDH = gaRKDH.run(v, c, C, best_pack_value, exact_solution)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generacja: 100/100 {duration:.2f}s")
    print("\n\n")
    all_fits = [best_fitnessRUSS, best_fitnessRUDH, best_fitnessRKSS, best_fitnessRKDH]
    all_sols = [best_solutionRUSS, best_solutionRUDH, best_solutionRKSS, best_solutionRKDH]
    all_errs = [gaRUSS.history_erros[gaRUSS.generations-1], gaRUDH.history_erros[gaRUDH.generations-1], gaRKSS.history_erros[gaRKSS.generations-1], gaRKDH.history_erros[gaRKDH.generations-1]]
    i = np.argmin(all_errs)
    best_solution = all_sols[i]
    best_fitness = all_fits[i]
    print(f"Najlepsze rozwiazanie genetyczne: {best_solution}")
    algo_names = ["RUSS", "RUDH", "RKSS", "RKDH"]
    best_solution_name = algo_names[i]
    
    # Porownanie rozwiazania genetycznego i dokladnego
    genetic_capacity = np.dot(best_solution, c)
    exact_value = best_pack_value
    exact_capacity = np.dot(exact_solution, c)
    print("Rodzaj algorytmu | Najlepsze przystosowanie | Koncowy blad | Zgodnosc bitowa")
    print(f"RUSS:              {best_fitnessRUSS}                    | {gaRUSS.history_erros[gaRUSS.generations-1]:.2f}%      | {gaRUSS.history_bit_comp[gaRUSS.generations-1]:.2f}%")
    print(f"RUDH:              {best_fitnessRUDH}                    | {gaRUDH.history_erros[gaRUDH.generations-1]:.2f}%      | {gaRUDH.history_bit_comp[gaRUDH.generations-1]:.2f}%")
    print(f"RKSS:              {best_fitnessRKSS}                    | {gaRKSS.history_erros[gaRKSS.generations-1]:.2f}%      | {gaRKSS.history_bit_comp[gaRKSS.generations-1]:.2f}%")
    print(f"RKDH:              {best_fitnessRKDH}                    | {gaRKDH.history_erros[gaRKDH.generations-1]:.2f}%      | {gaRKDH.history_bit_comp[gaRKDH.generations-1]:.2f}%")
    print(f"Wartosc rozwiazania dokladnego: {exact_value}")
    print(f"Najlepsze rozwiazanie genetyczne: {best_solution_name}")
    print(f"Stosunek wartosci najlepszego rozwiazania genetycznego do dokladnego: {best_fitness / exact_value:.2f}")
    ER = abs((best_fitness - exact_value)/exact_value) * 100
    print(f"Blad do rozwiazania dokladnego: {ER:.2f}%")
    print(f"Procentowa zgodnosc bitowa: {np.sum(best_solution == exact_solution) / n * 100:.2f}%")


    exact_plot = np.array([exact_value] * 100)
    plt.subplot(1,2,1)
    plt.plot(exact_plot, label = "wartosc dokladna", color = 'r')
    plt.plot(gaRUSS.history_fitness, label = "RUSS", color = 'b')
    plt.plot(gaRUDH.history_fitness, label = "RUDH", color = 'g')
    plt.plot(gaRKSS.history_fitness, label = "RKSS", color = 'y')
    plt.plot(gaRKDH.history_fitness, label = "RKDH", color = 'c')
    plt.xlabel('Generacja')
    plt.ylabel('Najlepsze przystosowanie')
    plt.title('Przystosowanie w trakcie pokolen')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(gaRUSS.history_erros, label = "RUSS", color = 'b')
    plt.plot(gaRUDH.history_erros, label = "RUDH", color = 'g')
    plt.plot(gaRKSS.history_erros, label = "RKSS", color = 'y')
    plt.plot(gaRKDH.history_erros, label = "RKDH", color = 'c')
    plt.xlabel('Generacja')
    plt.ylabel('Blad do dokladnej wartosci')
    plt.title('Blad w trakcie pokolen')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Rysowanie wykresu

    
