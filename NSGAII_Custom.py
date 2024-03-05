from typing import  List, TypeVar
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.solution import FloatSolution

import random
from typing import Generator, List, TypeVar


from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator,  MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")

class NSGAII_Custom(NSGAII):
    def __init__(self, problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(
            MultiComparator([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])
        ),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        dominance_comparator: Comparator = store.default_comparator,
        swap_probability = 0.8):
        super().__init__(
                        problem,
                        population_size,
                        offspring_population_size,
                        mutation,
                        crossover,
                        selection,
                        termination_criterion = store.default_termination_criteria,
                        population_generator = store.default_generator,
                        population_evaluator = store.default_evaluator,
                        dominance_comparator = store.default_comparator
                        )
        self.swap_probability = swap_probability

    def create_initial_solutions(self) -> List[S]:
            return [self.problem.generate_sparse_random_instance_fi() for _ in range(self.population_size)]

    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)

        if self.evaluations < self.termination_criterion.max_evaluations - self.population_size:
            offspring_population = self.reset_offspring(offspring_population)

        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)


    # def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
    #         """This method joins the current and offspring populations to produce the population of the next generation
    #         by applying the ranking and crowding distance selection.
    #
    #         :param population: Parent population.
    #         :param offspring_population: Offspring population.
    #         :return: New population after ranking and crowding distance selection is applied.
    #         """
    #
    #         offspring_population = self.reset_offspring(offspring_population)
    #
    #         ranking = FastNonDominatedRanking(self.dominance_comparator)
    #         density_estimator = CrowdingDistance()
    #
    #         r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
    #         solutions = r.replace(population, offspring_population)
    #
    #         return solutions

    def reset_offspring(self, offspring):
        updated_offspring = []

        for candidate in offspring:
                c = candidate.variables
                for i, _ in enumerate(c):
                      
                        swap_probability = random.random()

                        if(swap_probability > self.swap_probability):
                                c[i] = self.problem.data_instance[i]

                new_solution = FloatSolution(
                        lower_bound=self.problem.lower_bound,
                        upper_bound=self.problem.upper_bound,
                        number_of_objectives=self.problem.number_of_objectives(),
                        number_of_constraints=self.problem.number_of_constraints())
                
                new_solution.variables = c

                updated_offspring.append(new_solution)
            
        return updated_offspring