import numpy as np
import random
import matplotlib.pyplot as plt
import time

class GeneticAlgorithm:
    def __init__(self, n, fitness_function, population_size=1000, mutation_rate=0.01, crossover_rate=0.9, generations=100,
                 selection_type="roulette", crossover_type="single", penalty_type="soft"):
        #Inicjalizuje algorytm genetyczny.

        self.n = n # liczba genow w chromosomie
        self._fitness = fitness_function # Wskaznik na funkcje przystosowania
        self.population_size = population_size # Liczba osobnikow
        self.mutation_rate = mutation_rate # Prawdopodobienstwo na mutacje w bicie
        self.crossover_rate = crossover_rate # Prawdopodobienstwo krzyzowania w parze rodzicow
        self.generations = generations # Liczba pokolen (iteracji)
        self.selection_type = selection_type # Selekcja "roulette", "rank" lub "eugenic" xdd
        self.crossover_type = crossover_type # Sposob krzyzowania "single" lub "double"
        self.penalty_type = penalty_type # Rodzaj kary "soft" lub "hard"
        self.population = self._initialize_population()
        self.history_fitness = []  # Przystosowanie najlepszego osobnika kazdej populacji - do wykresu
        self.history_erros = []  # Przystosowanie najlepszego osobnika kazdej populacji - do wykresu
        self.history_bit_comp = []  # Przystosowanie najlepszego osobnika kazdej populacji - do wykresu
        
        self.weights = [] # Pomocnicze - optymalizacja

    def _initialize_population(self):
        # Losowe wartosci 0 lub 1 w kazdym chromosomie poczatkowej populacji
        return np.random.randint(0, 2, size=(self.population_size, self.n))

    def _select_parents(self, v, c, C):
        # Wybiera 2 rodzicow z populacji zgodnie z rodzajem wybranej selekcji
        if self.selection_type == "roulette":
            parents = random.choices(self.population, self.probabilitiesRU, k = 2)
            i = 0 # Ograniczenie zeby nie utknal jak jest duzo elementow powtarzajacych sie
            while np.array_equal(parents[0], parents[1]) and i < 3:
                i+=1
                parents = random.choices(self.population, self.probabilitiesRU, k = 2)
            return parents
        
        elif self.selection_type == "rank":
            # Selekcja rankingowa
            parents = random.choices(self.population, self.probabilitiesRK, k=2)
            i = 0 # Ograniczenie zeby nie utknal jak jest duzo elementow powtarzajacych sie
            while np.array_equal(parents[0], parents[1]) and i < 3:
                i+=1
                parents = random.choices(self.population, self.probabilitiesRK, k = 2)
            return parents
        
        elif self.selection_type == "eugenic":
            # Zawsze wybiera 2 najlepszych rodzicow
            return self.population[0], self.population[1]

    def _crossover(self, parent1, parent2):
        # Krzyzowanie genow rodzicow
        if random.random() < self.crossover_rate:
            if self.crossover_type == "single":
                crossover_point = random.randint(1, self.n - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                return child1, child2
            elif self.crossover_type == "double":
                crossover_point1 = random.randint(1, self.n - 1)
                crossover_point2 = random.randint(crossover_point1 + 1, self.n)
                child1 = np.concatenate((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
                child2 = np.concatenate((parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
                return child1, child2
        else:
            return parent1, parent2

    def _mutate(self, individual):
        # Losowa zmiana wartosci genu
        return np.array([1 - gene if random.random() < self.mutation_rate else gene for gene in individual])

    def run(self, v, c, C, exact_best_value, exact_solution):
        best_solution = None
        best_fitness = -1.0
        
        for generation in range(self.generations):
            time_start = time.time()
            ###############
            print(f"Generacja: {generation + 1}/{self.generations}", end="\r")
            new_population = []
            
            self.weights = np.array([self._fitness(individual, v, c, C, self.penalty_type) for individual in self.population])
            sorted_indices = np.argsort(self.weights)[::-1]
            self.population = self.population[sorted_indices]
            
            if(self.selection_type == "roulette"):
                self.total_fitness = sum(self.weights)
                self.probabilitiesRU = self.weights / self.total_fitness
            elif (self.selection_type == "rank"):
                m = len(self.population)
                ranks = np.arange(m, 0, -1)
                self.probabilitiesRK = (2 * ranks)/(m * (m + 1)) 

            # Tworzenie nowej populacji
            for i in range(self.population_size // 2):
                parent1, parent2 = self._select_parents(v, c, C)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1))
                new_population.append(self._mutate(child2))
            
            self.population = np.array(new_population)

            # Sprawdzanie najlepszego rozwiazania w poprzedniej populacji
            for individual in self.population:
                fitness = self._fitness(individual, v, c, C, self.penalty_type)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual
            
            #print(f"Najlepsze przystosowanie: {best_fitness}")
            ER = abs((best_fitness - exact_best_value)/exact_best_value) * 100
            #print(f"Blad do rozwiazania dokladnego: {ER:.2f}%")
            #print(best_solution)
            #print(exact_solution)
            ER_bit = np.sum(best_solution == exact_solution) / len(exact_solution) * 100
            #print(f"Zgodnosc bitowa z rozwiazaniem dokladnym: {ER_bit:.2f}%")
            time_end = time.time()
            duration = time_end - time_start
            #print(f"Czas trwania generacji: {duration:.2f}s\n\n")
            
            self.history_fitness.append(best_fitness)  # Zapisz przystosowanie na tej generacji
            self.history_erros.append(ER)
            self.history_bit_comp.append(ER_bit)
        
        return best_solution, best_fitness

        
# Funkcja przystosowania
def fitness_function(individual, v, c, C, penalty_type):
    # Wartosci funkcji przystosowania z uwzglednieniem kary za przekroczenie max pojemnosci
    value = np.dot(individual, v)
    weight = np.dot(individual, c)
    if weight > C:
        if penalty_type == "soft":
            penalty = abs(1.0 * weight/C)-1.0  # Miekka kara - tyle o ile przekroczyl
        elif penalty_type == "hard":
            return 0 # Drastyczna kara
        return int(value - value * penalty)
    return value