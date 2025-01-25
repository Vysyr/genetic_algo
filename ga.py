import numpy as np
import random
import matplotlib.pyplot as plt
import time

class GeneticAlgorithm:
    def __init__(self, n, fitness_function, population_size=1000, mutation_rate=0.01, crossover_rate=0.9, generations=100,
                 selection_type="roulette", crossover_type="single", penalty_type="soft"):
        #Inicjalizuje algorytm genetyczny.
        
        #:param n: Rozmiar problemu (liczba genow w chromosomie)
        #:param fitness_function: Funkcja przystosowania (wskaŸnik na funkcjê)
        #:param population_size: Rozmiar populacji
        #:param mutation_rate: Prawdopodobienstwo mutacji (na poziomie bitu)
        # :param crossover_rate: Prawdopodobienstwo krzyzowania (na poziomie pary rodzicow)
        # :param generations: Liczba pokolen
        # :param selection_type: Typ selekcji ("roulette" lub "rank")
        # :param crossover_type: Typ krzyzowania ("single" lub "double")
        # :param penalty_type: Typ kary ("soft" lub "hard")
        self.n = n
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.penalty_type = penalty_type
        self.population = self._initialize_population()
        self.history = []  # Do przechowywania przystosowania w ka¿dej generacji

    def _initialize_population(self):

        return np.random.randint(0, 2, size=(self.population_size, self.n))

    def _fitness(self, individual, v, c, C):

        value = np.dot(individual, v)
        weight = np.dot(individual, c)
        if weight > C:
            if self.penalty_type == "soft":
                penalty = (weight - C)  # Miêkka kara
            elif self.penalty_type == "hard":
                penalty = (weight - C) * 10 # Drastyczna kara
            return value - penalty
        return value

    def _select_parents(self, v, c, C):

        if self.selection_type == "roulette":
            weights = [self._fitness(individual, v, c, C) for individual in self.population]
            total_fitness = sum(weights)
            probabilities = [w / total_fitness for w in weights]
            return random.choices(self.population, probabilities, k=2)
        
        elif self.selection_type == "rank":
            # Selekcja rankingowa
            fitness_values = np.array([self._fitness(individual, v, c, C) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)[::-1]
            sorted_population = self.population[sorted_indices]
            parents = sorted_population[:2]
            return parents[0], parents[1]

    def _crossover(self, parent1, parent2):
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
        return np.array([1 - gene if random.random() < self.mutation_rate else gene for gene in individual])

    def run(self, v, c, C, exact_best_value, exact_solution):
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            time_start = time.time()
            print(f"Generacja: {generation + 1}/{self.generations}")
            new_population = []
            
            # Tworzenie nowej populacji
            for _ in range(self.population_size // 2):
                parent1, parent2 = self._select_parents(v, c, C)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1))
                new_population.append(self._mutate(child2))
            
            self.population = np.array(new_population)
            
            # Sprawdzanie najlepszego rozwiazania w biezacej populacji
            for individual in self.population:
                fitness = self._fitness(individual, v, c, C)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual
            self.history.append(best_fitness)  # Zapisz przystosowanie na tej generacji
            print(f"Najlepsze przystosowanie: {best_fitness}")
            ER = abs((best_fitness - exact_best_value)/exact_best_value) * 100
            print(f"Blad do rozwiazania dokladnego: {ER:.2f}%")
            #print(best_solution)
            #print(exact_solution)
            ER_bit = np.sum(best_solution == exact_solution) / len(exact_solution) * 100
            print(f"Zgodnosc bitowa z rozwiazaniem dokladnym: {ER_bit:.2f}%")
            time_end = time.time()
            duration = time_end - time_start
            print(f"Czas trwania generacji: {duration:.2f}s\n\n")
        
        return best_solution, best_fitness

    def plot_fitness(self):
        plt.plot(self.history)
        plt.xlabel('Generacja')
        plt.ylabel('Najlepsze przystosowanie')
        plt.title('Przystosowanie w trakcie pokolen')
        plt.show()
        
# Funkcja przystosowania dla plecaka
def fitness_function(individual, v, c, C):
    return np.dot(individual, v) if np.dot(individual, c) <= C else 0