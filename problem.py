import decimal
import heapq
import random
import numpy as np

import random
import pandas as pd
# import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from disagreement import Disagreement


class CounterfactualConsensus(FloatProblem):

    def __init__(self,
                 data_instance,
                 base_counterfactuals,
                 categorical_features_idxs,
                 immutable_features_idxs,
                 continuous_features_idxs,
                 data,
                 predict_fn,
                 predict_proba_fn,
                 disagreement_method,
                 seed,
                 parallel=False,
                 wachter=False,
                 # feature_ranges,
                 # class_labels,
                 # labels=[],
                 # feature_entropy=0.25,
                 # sparse=False,
                 ):
        super().__init__()
        self.data_instance = data_instance
        self.base_counterfactuals = base_counterfactuals
        self.categorical_features_idxs = categorical_features_idxs
        self.immutable_features_idxs = immutable_features_idxs
        # self.immutable_features_set = set(immutable_features_idxs)
        self.continuous_features_idxs = continuous_features_idxs
        self.data = data
        self.predict_fn = predict_fn
        # predict_proba_fn,
        self.disagreement_method = disagreement_method
        self.seed = seed
        # self.results_manager = ResultsManager()
        self.parallel = parallel
        self.wachter = wachter

        self.bounds = self.get_bounds()
        self.feature_entropy = 0.25
        self.target_class = 1 - predict_fn(data_instance)  # TODO don't assume binary classification
        self.feature_ranges = self.get_bounds()

        # self.labels = labels if labels else list(range(1, len(base_counterfactuals) + 1))
        # self.class_labels = class_labels
        # self.num_generations = num_generations
        # self.categories = self.get_categories(categorical_features_idxs)
        # self.feature_ranges = self.get_feature_ranges(feature_ranges)

        self.disagreement = Disagreement(
            disagreement_method=self.disagreement_method,
            data_instance=self.data_instance,
            base_counterfactuals=self.base_counterfactuals,
            instance_probability=predict_proba_fn(data_instance),
            predict_proba_fn=predict_proba_fn,
            target_class=self.target_class,
            bounds=self.bounds,
            parallel=self.parallel
        )
        # self.is_counterfactual_valid = self.utils.is_counterfactual_valid
        # self.print_results = self.utils.print_results

        self.obj_labels = ['Proximity', 'Sparsity', 'Disagreement']
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.lower_bound = [self.feature_ranges[i][0] for i in range(len(self.feature_ranges))]
        self.upper_bound = [self.feature_ranges[i][1] for i in range(len(self.feature_ranges))]

        output_columns = [i for i in range(self.number_of_variables())] + [obj for obj in self.obj_labels]
        self.all_solutions = self.init_output_dataframe()
        self.candidate_instances = self.init_output_dataframe()

    def init_output_dataframe(self):
        df = pd.DataFrame(columns=[i for i in range(self.number_of_variables())] + [obj for obj in self.obj_labels])
        return df

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def number_of_constraints(self) -> int:
        return 0

    # def get_feature_idxs(self, type_of_features: list) -> list:  # Tiwonge
    #     all_features = list(self.data.columns)
    #     return [all_features.index(f) for f in all_features if f in type_of_features]

    # def __str__(self):
    #     attributes_str = [
    #         f"categorical_features: {self.categorical_features}",
    #         f"immutable_features: {self.immutable_features}",
    #         f"population_size: {self.population_size}",
    #         f"num_generations: {self.num_generations}",
    #         f"target_class: {self.target_class}",
    #         f"categories: {self.categories}",
    #         f"feature_ranges: {self.feature_ranges}",
    #     ]
    #     return "\n".join(attributes_str)
    #
    # def to_string(self):
    #     return str(self)

    def get_bounds(self):  # Craig original
        data = self.data.to_numpy()  # Tiwonge converted to numpy
        feature_ranges = []

        for i in range(len(self.data_instance)):
            lower_bound = min(data[:, i])
            upper_bound = max(data[:, i])
            feature_ranges.append((lower_bound, upper_bound))
        return feature_ranges

    def create_solution(self) -> FloatSolution:
        """Creates a jmetalpy FloatSolution"""
        new_solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())
        new_solution.variables = [random.uniform(self.lower_bound[i],
                                                 self.upper_bound[i]) for i in range(self.number_of_variables())]
        return new_solution

    def get_candidate_instance(self, solution: FloatSolution) -> list[float]:
        """Converts a jmetal FloatSolution into a candidate instance"""
        candidate_instance = []

        for idx, value in enumerate(solution.variables):
            if idx in self.immutable_features_idxs:
                candidate_instance.insert(idx, self.data_instance[idx])

            elif idx in self.categorical_features_idxs:
                category = self.get_category(idx, value)
                candidate_instance.insert(idx, category)

            elif idx in self.continuous_features_idxs:
                candidate_instance.insert(idx, value)

            else:
                raise TypeError(f"solution variable: {idx} is not properly defined")
        return candidate_instance

    def get_category(self, idx: int, value: float) -> float | None:
        """Places the value of categorical feature into its corresponding category"""
        category = None
        if idx in self.categorical_features_idxs:
            categories = sorted(self.data.iloc[:, idx].unique())
            category_max = 0
            for i in range(len(categories)):
                category_min = category_max
                category_max += 1 / len(categories)
                if category_min <= value < category_max:
                    category = categories[i]
            return category
        else:
            raise TypeError(f"solution variable: {idx} is not defined as categorical variable")

    def get_categories(self, categorical_features):  # Craig original
        categories = {}
        data = self.data.to_numpy()  # Tiwonge converted to numpy
        for feature in categorical_features:
            options = np.unique(data[:, feature])
            categories[feature] = options
        return categories

    def evaluate(self, solution: FloatSolution):
        candidate_instance = self.get_candidate_instance(solution)

        prediction = self.predict_fn(candidate_instance)
        if prediction == self.target_class:

            proximity_score = self.disagreement.calculate_proximity(self.data_instance, candidate_instance, True)
            sparsity_score, _ = self.disagreement.calculate_sparsity(candidate_instance)
            disagreement_score = self.disagreement.calculate_average_disagreement(candidate_instance)

            # if self.wachter:
            #     penalty_score = self.disagreement.misclassification_penalty_wachter(candidate_instance)
            # else:
            #     penalty_score = self.disagreement.misclassification_penalty(candidate_instance)

            solution.objectives[0] = proximity_score
            solution.objectives[1] = sparsity_score
            solution.objectives[2] = disagreement_score
            # solution.objectives[3] = penalty_score
        else:
            penalty_score = 2
            solution.objectives = [penalty_score for _ in range(self.number_of_objectives())]

        self.record_solution(solution)
        self.record_candidate_instance(candidate_instance, solution)
        return solution

    def record_solution(self, solution: FloatSolution):
        evaluated_solution = [var for var in solution.variables] + [obj for obj in solution.objectives]
        self.all_solutions.loc[len(self.all_solutions.index)] = evaluated_solution

    def record_candidate_instance(self, candidate_instance: list[float], solution:FloatSolution):
        candidate = [i for i in candidate_instance] + [obj for obj in solution.objectives]
        self.candidate_instances.loc[len(self.candidate_instances.index)] = candidate

    def name(self) -> str:
        return 'CFC'
