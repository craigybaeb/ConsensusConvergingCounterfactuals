import random
import numpy as np

import random
import pandas as pd
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
                 feature_importances,
                 sparsity_probability=0.5
                 ):
        super().__init__()
        self.data_instance = data_instance
        self.base_counterfactuals = base_counterfactuals
        self.categorical_features_idxs = categorical_features_idxs
        self.immutable_features_idxs = immutable_features_idxs
        self.immutable_features_set = set(immutable_features_idxs)
        self.continuous_features_idxs = continuous_features_idxs
        self.sparsity_probability = sparsity_probability
        self.sparsity_probabilities = feature_importances
        self.data = data
        self.like_data = self.get_like_data(predict_fn)
        self.predict_fn = predict_fn
        self.disagreement_method = disagreement_method
        self.seed = seed
        self.bounds = self.get_bounds()
        self.feature_entropy = 0.25
        self.target_class = 1 - predict_fn(data_instance)  # TODO don't assume binary classification
        self.feature_ranges = self.get_bounds()

        self.disagreement = Disagreement(
            disagreement_method=self.disagreement_method,
            data_instance=self.data_instance,
            base_counterfactuals=self.base_counterfactuals,
            instance_probability=predict_proba_fn(data_instance),
            predict_proba_fn=predict_proba_fn,
            target_class=self.target_class,
            bounds=self.bounds,
            categorical_features_idxs=self.categorical_features_idxs,
            continuous_feature_ranges=self.continuous_features_idxs,
            feature_ranges=self.feature_ranges
        )

        self.obj_labels = ['Proximity', 'Sparsity', 'Disagreement', 'Plausibility']
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.lower_bound = [self.feature_ranges[i][0] for i in range(len(self.feature_ranges))]
        self.upper_bound = [self.feature_ranges[i][1] for i in range(len(self.feature_ranges))]

        output_columns = [i for i in range(self.number_of_variables())] + [obj for obj in self.obj_labels]
        self.all_solutions = self.init_output_dataframe()
        self.candidate_instances = self.init_output_dataframe()

    def init_output_dataframe(self):
        df = pd.DataFrame(columns=[i for i in range(self.number_of_variables())] + [obj for obj in self.obj_labels])
        return df

    def generate_sparse_random_instance(self):
        candidate = np.array(self.generate_random_instance()).astype(float)

        for i, _ in enumerate(candidate):
            swap_probability = random.random()

            if(swap_probability > self.sparsity_probability):
                candidate[i] = self.data_instance[i]

        new_solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())
        
        new_solution.variables = candidate
        
        return new_solution
    
    def generate_sparse_random_instance_fi(self):
        candidate = np.array(self.generate_random_instance()).astype(float)

        for i, _ in enumerate(candidate):
            swap_probability = self.sparsity_probabilities[i]

            if(swap_probability > self.sparsity_probability):
                candidate[i] = self.data_instance[i]

        new_solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())
        new_solution.variables = candidate
        
        return new_solution
    
    def generate_random_instance(self):
        candidate_instance = []

        for i, _ in enumerate(self.feature_ranges):
            if i in self.immutable_features_set:
                # For immutable features, use the original value
                candidate_instance.append(self.data_instance[i])
            elif i in self.categorical_features_idxs:
                possible_values = sorted(set(int(data[i]) for data in np.array(self.data)))
                candidate_instance.append(random.choice(possible_values))
            else:
                candidate_value = random.uniform(self.lower_bound[i],
                                                 self.upper_bound[i])
                
                candidate_value = self.format_to_original_precision(self.data_instance[i], candidate_value)

                candidate_instance.append(candidate_value)

        return candidate_instance


    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def number_of_constraints(self) -> int:
        return 0
    
    def get_like_data(self, predict_fn):
        """
        Returns data similar to the input data based on class similarity using a given model.

        Parameters:
        - input_data: The input data for which similar data is to be found. Should be in the same format as the dataset used to train the model.
        - dataset: The dataset from which similar data is to be retrieved. Assumes the last column is the class label.
        - model: A trained model capable of making class predictions.

        Returns:
        - similar_data: A subset of the dataset containing only instances of the class predicted for the input_data.
        """

        data = self.data.to_numpy()
        # Predict the class of the input data
        predicted_class = predict_fn(self.data_instance)

        # Separate features and labels from the dataset
        features = data[:, :-1]  # Assuming all columns except the last one are features
        labels = data[:, -1]  # Assuming the last column is the label

        # Filter the dataset to include only instances of the predicted class
        similar_data = features[labels == predicted_class]

        return similar_data



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
                formatted_value = self.format_to_original_precision(self.data_instance[idx], value)
                candidate_instance.insert(idx, formatted_value)
  

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
                if category_min <= value <= category_max:
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
    
    def format_to_original_precision(self, original, new_float):
        # Convert the original float to a string to find the decimal precision
        original_str = str(original)
        
        # Check if there is a decimal point in the original number
        if '.' in original_str:
            decimal_precision = len(original_str.split('.')[1])
        else:
            decimal_precision = 0  # No decimal places if no decimal point
        
        # Format the new float to match the original precision
        formatted_new_float = round(new_float, decimal_precision)
        
        return formatted_new_float

    def evaluate(self, solution: FloatSolution):
        candidate_instance = self.get_candidate_instance(solution)

        prediction = self.predict_fn(candidate_instance)
        if prediction == self.target_class:

            proximity_score = self.disagreement.calculate_proximity(self.data_instance, candidate_instance, True)
            sparsity_score, _ = self.disagreement.calculate_sparsity(candidate_instance)
            disagreement_score = self.disagreement.calculate_average_disagreement(candidate_instance)
            plausibility_score = self.disagreement.calculate_plausibility(candidate_instance, self.data)

            solution.objectives[0] = proximity_score
            solution.objectives[1] = sparsity_score
            solution.objectives[2] = disagreement_score
            solution.objectives[3] = plausibility_score
            # solution.objectives[4] = penalty_score
            
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
