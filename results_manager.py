import numpy as np
import pandas as pd
import os
import datetime
from jmetal.util.solution import print_function_values_to_file


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
        self.candidate_instances = target_problem.candidate_instances

    def set_results_folder(self):
        folder = f'{self.base_folder}/{self.problem_name}/{self.solver_name}_{self.time_now()}/seed_{self.rseed}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def save_pareto_solutions(self, front) -> None:
        solutions_df = pd.DataFrame(columns=self.target_problem.all_solutions.columns)
        for solution in front:
            row = np.concatenate((solution.variables,solution.objectives), axis=None)  # add constraints later
            solutions_df.loc[len(solutions_df)] = row
        solutions_df.to_csv(
            path_or_buf=f'{self.destination_path}/{self.problem_name}_{self.solver_name}_pareto_solutions.csv',
            index=False)

    def save_function_values(self, front):
        print_function_values_to_file(
            solutions=front,
            filename=f'{self.destination_path}/{self.problem_name}_{self.solver_name}_pareto_function_values_seed_{self.rseed}.pf')

    def save_candidate_instances(self):
        self.candidate_instances.to_csv(
            path_or_buf=f"{self.destination_path}/{self.problem_name}_{self.solver_name}_candidate_instances.csv",
            index=False)

    def save_all_solutions(self):
        self.all_solutions.to_csv(
            path_or_buf=f"{self.destination_path}/{self.problem_name}_{self.solver_name}_all_solutions.csv",
            index=False)

    def save_results(self, front):
        self.save_pareto_solutions(front)
        self.save_function_values(front)
        self.save_all_solutions()
        self.save_candidate_instances()

    @staticmethod
    def time_now():
        ct = datetime.datetime.now()
        ts = ct.timestamp()
        date_time = datetime.datetime.fromtimestamp(ts, tz=None)
        return date_time.strftime("%d-%m-%Y_%H-%M")