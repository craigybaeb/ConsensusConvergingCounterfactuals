import pandas as pd
# import csv
import os
import datetime
# from jmetal.core.solution import FloatSolution
from jmetal.util.solution import print_function_values_to_file
# from utils.core_functions import time_now


class ResultsManager:
    def __init__(self, target_problem, solver_name, rseed, base_folder):
    # def __init__(self, problem_name, solver_name, rseed, base_folder):
        self.target_problem = target_problem
        self.problem_name = target_problem.name()
        self.solver_name = solver_name
        self.rseed = rseed
        self.base_folder = base_folder
        self.destination_path = self.set_results_folder()
        self.all_solutions = target_problem.all_solutions

    def set_results_folder(self):
        folder = f'{self.base_folder}/{self.problem_name}/{self.solver_name}_{self.time_now()}/seed_{self.rseed}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def save_pareto_solutions(self, front) -> None:
        solutions_df = pd.DataFrame(columns=self.target_problem.all_solutions.columns)
        for solution in front:
            row = solution.variables + solution.objectives  # add constraints later
            solutions_df.loc[len(solutions_df)] = row
        solutions_df.to_csv(
            path_or_buf=f'{self.destination_path}/{self.problem_name}_{self.solver_name}_pareto_solutions.csv',
            index=False)

    def save_function_values(self, front):
        print_function_values_to_file(
            solutions=front,
            filename=f'{self.destination_path}/{self.problem_name}_{self.solver_name}_pareto_function_values_seed_{self.rseed}.pf')

    def save_all_solutions(self):
        self.all_solutions.to_csv(
            path_or_buf=f"{self.destination_path}/{self.problem_name}_{self.solver_name}_all_solutions.csv",
            index=False)

    @staticmethod
    def time_now():
        ct = datetime.datetime.now()
        ts = ct.timestamp()
        date_time = datetime.datetime.fromtimestamp(ts, tz=None)
        return date_time.strftime("%d-%m-%Y_%H-%M")