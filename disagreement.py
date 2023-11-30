import numbers
import numpy as np
from scipy.spatial.distance import cityblock, cosine, euclidean
from functools import lru_cache
import concurrent.futures


class Disagreement:
    """
    Disagreement class for calculating disagreement measures between instances and counterfactuals.

    @param disagreement_method: The method for calculating disagreement.
        Possible values: "feature_overlap", "cosine_distance", "euclidean_distance", "manhattan_distance.
    @type disagreement_method: str

    @param data_instance: The original data instance for which counterfactuals are being generated.
    @type data_instance: list of int

    @param base_counterfactuals: The base counterfactual instances.
    @type base_counterfactuals: list of list of int

    @param categorical_features: List of indices for categorical features in the data.
    @type categorical_features: list of int

    @param continuous_feature_ranges: List of tuples representing the ranges for continuous features.
    @type continuous_feature_ranges: list of (int, int)

    @param predict_fn: The function used for making predictions with the model.
    @type predict_fn: callable

    @param predict_proba_fn: The function used for making probability predictions with the model.
    @type predict_proba_fn: callable

    @param target_class: The target class for generating counterfactuals.
    @type target_class: int

    @param feature_ranges: List of tuples representing the ranges for each feature in the data.
    @type feature_ranges: list of (int, int)
    """

    def __init__(self,
                 disagreement_method,
                 data_instance,
                 base_counterfactuals,
                 instance_probability,
                 predict_proba_fn,
                 target_class,
                 bounds,
                 parallel,
                 # categorical_features_idxs,
                 # continuous_feature_ranges,
                 # predict_fn,
                 # feature_ranges,
                 ):
        self.data_instance = data_instance
        self.base_counterfactuals = base_counterfactuals
        self.instance_probability = instance_probability
        self.predict_proba_fn = predict_proba_fn
        self.target_class = target_class
        self.bounds = bounds
        self.parallel = parallel
        # self.categorical_features = categorical_features_idxs
        # self.continuous_feature_ranges = continuous_feature_ranges
        # self.predict_fn = predict_fn
        # self.feature_ranges = feature_ranges

        self.calculate_disagreement = self.set_disagreement_method(disagreement_method)
        self.normalized_instance = self.normalize_instance(data_instance)
        self.proxmity_weight = 0.3
        self.overlap_weight = 0.7

    #     def __str__(self):
    #         """
    #         Return a string representation of the Disagreement object.
    #
    #         @return: String representation of the Disagreement object.
    #         @rtype: str
    #         """
    #         predict_fn_name = self.predict_fn.__name__ if callable(self.predict_fn) else "N/A"
    #         predict_proba_fn_name = self.predict_proba_fn.__name__ if callable(self.predict_proba_fn) else "N/A"
    #
    #         return f"Disagreement Object:\n" \
    #                f"Disagreement Method: {self.calculate_disagreement.__name__}\n" \
    #                f"Data Instance: {self.data_instance}\n" \
    #                f"Base Counterfactuals: {self.base_counterfactuals}\n" \
    #                f"Categorical Features: {self.categorical_features}\n" \
    #                f"Continuous Feature Ranges: {self.continuous_feature_ranges}\n" \
    #                f"Target Class: {self.target_class}\n" \
    #                f"Feature Ranges: {self.feature_ranges}\n" \
    #                f"Predict Function Name: {predict_fn_name}\n" \
    #                f"Predict Proba Function Name: {predict_proba_fn_name}\n" \
    #                f"Normalized Instance: {self.normalized_instance}\n"
    #
    # def to_string(self):
    #     """
    #     Convert the Disagreement object to a string.
    #
    #     @return: String representation of the Disagreement object.
    #     @rtype: str
    #     """
    #     return self.__str__()

    def euclidean_distance(self, instance1, instance2):
        """
        Calculate the Euclidean distance between two instances.

        @param instance1: The first instance.
        @type instance1: list of int or float

        @param instance2: The second instance.
        @type instance2: list of int or float

        @return: The Euclidean distance between the instances.
        @rtype: float
        """
        self.validate_instances(instance1, instance2)

        normalised_instance1 = np.array(self.normalize_instance(instance1))
        normalised_instance2 = np.array(self.normalize_instance(instance2))

        return euclidean(normalised_instance1, normalised_instance2)

    def calculate_cosine_distance(self, instance1, instance2):
        """
        Calculate the cosine distance between two counterfactual instances.

        @param instance1: The first data instance.
        @type instance1: list of int or float

        @param instance2: The second data instance.
        @type instance2: list of int or float

        @return: The cosine distance between the counterfactual instances.
        @rtype: float
        """
        self.validate_instances(instance1, instance2)

        normalised_instance1 = self.normalize_instance(instance1)
        normalised_instance2 = self.normalize_instance(instance2)

        return cosine(normalised_instance1, normalised_instance2)

    def calculate_manhattan_distance(self, instance1, instance2):
        """
        Calculate the manhattan distance between two instances.

        @param instance1: The first data instance.
        @type instance1: list of int or float

        @param instance2: The second datainstance.
        @type instane2: list of int or float

        @return: The manhattan distance between the counterfactual instances.
        @rtype: float
        """
        self.validate_instances(instance1, instance2)

        normalised_instance1 = self.normalize_instance(instance1)
        normalised_instance2 = self.normalize_instance(instance2)

        return cityblock(normalised_instance1, normalised_instance2)

    # TODO fix or remove
    @lru_cache(maxsize=None)
    def median_absolute_deviation(self, data):
        """
        Calculate the median absolute deviation of the data.

        @param data: The data for which to calculate the median absolute deviation.
        @type data: list of float

        @return: The median absolute deviation of the data.
        @rtype: float
        """
        data_tuple = tuple(data)
        median = np.median(data)
        absolute_deviations = np.abs(np.array(data) - median)

        return np.median(absolute_deviations)

    def normalize_instance(self, instance):
        """
        Normalize the given instance using min-max scaling.

        @param instance: The instance to be normalized.
        @type instance: list of float

        @return: The normalized instance.
        @rtype: list of float
        """
        normalized_instance = []

        for i, (min_val, max_val) in enumerate(self.bounds):
            # Make sure the feature range is valid
            if min_val > max_val:
                raise ValueError("Invalid feature range: min_val must be less than or equal to max_val.")

            # Normalize the feature value using min-max scaling
            feature_value = instance[i]
            if min_val == max_val:
                normalized_instance.append(feature_value)
            else:
                normalized_value = (feature_value - min_val) / (max_val - min_val)
                normalized_instance.append(normalized_value)

        return normalized_instance

    def calculate_proximity(self, data_instance, counterfactual_instance, normalise=False):
        """
        Calculate the proximity score between a data instance and a counterfactual instance.

        @param data_instance: The data instance.
        @type data_instance: list of float

        @param counterfactual_instance: The counterfactual instance.
        @type counterfactual_instance: list of float

        @param normalise: Flag indicating whether to normalize the instances before calculating proximity.
        @type normalise: bool

        @return: The proximity score between the instances.
        @rtype: float
        """
        query_instance = data_instance
        counterfactual = counterfactual_instance

        if normalise:
            query_instance = self.normalized_instance
            counterfactual = self.normalize_instance(counterfactual_instance)

        l1_norm_score = self.calculate_manhattan_distance(query_instance, counterfactual)

        return l1_norm_score / len(data_instance)

    def calculate_feature_overlap(self, candidate_instance, base_cf):
        """
        Calculate the feature overlap score between a candidate instance and a base counterfactual.

        @param candidate_instance: The candidate instance.
        @type candidate_instance: list of float

        @param base_cf: The base counterfactual.
        @type base_cf: list of float

        @return: The feature overlap score between the instances.
        @rtype: float
        """
        cf1_changed_features = set(
            i for i, cf1_val in enumerate(candidate_instance) if cf1_val != self.data_instance[i])
        cf2_changed_features = set(i for i, cf2_val in enumerate(base_cf) if cf2_val != self.data_instance[i])

        union_changed_features = cf1_changed_features.union(cf2_changed_features)
        total_features = len(self.data_instance)
        agreement_score = abs(len(union_changed_features)) / total_features

        return 1 - agreement_score

    def set_disagreement_method(self, function):
        """
        Set the disagreement method based on the provided input.

        @param function: The method for calculating disagreement.
        @type function: str

        @return: The selected disagreement method function.
        @rtype: function
        """
        if function == "feature_overlap":
            return self.calculate_feature_overlap
        elif function == "cosine_distance":
            return self.calculate_cosine_distance
        elif function == "euclidean_distance":
            return self.euclidean_distance
        elif function == "manhattan_distance":
            return self.calculate_manhattan_distance
        elif function == "direction_overlap":
            return self.calculate_direction_overlap
        else:
            # Default to manhattan distance if an invalid method is provided
            return self.calculate_manhattan_distance

    def calculate_average_disagreement(self, candidate_instance):  # Tiwonge updated. Moved over from evaluation
        """
        *****Updated to compare candidate instance with base counterfactuals, not the other way round.*****

        Calculate the average disagreement score between candidate instances and a base counterfactual.

        @param candidate_instance: candidate instances.
        @type candidate_instance: list of float

        # @param base_cf: The base counterfactual instance.
        # @type base_cf: list of float

        @return: The average disagreement score.
        @rtype: float
        """
        base_cf_scores = []

        if self.parallel:
            # Use ThreadPoolExecutor to compute agreement scores in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agreement_scores = list(
                    executor.map(self.calculate_agreement_score, self.base_counterfactuals, [candidate_instance] * len(self.base_counterfactuals)))

            base_cf_scores = agreement_scores  # Store the agreement scores in the base_cf_scores list
        else:
            for base_counterfactual in self.base_counterfactuals:
                agreement_score = self.calculate_agreement_score(candidate_instance, base_counterfactual)
                base_cf_scores.append(agreement_score)

        return sum(base_cf_scores) / len(base_cf_scores)

    def calculate_agreement_score(self, candidate_instance, base_cf):  # Tiwonge updated t match updated calculate_base_cf_scores. Moved over from evaluation
        """
        Calculate the disagreement score between a candidate instance and a base counterfactual.

        @param candidate_instance: The candidate instance.
        @type candidate_instance: list of float

        @param base_cf: The base counterfactual instance.
        @type base_cf: list of float

        @return: The disagreement score.
        @rtype: float
        """
        # agreement_score = self.disagreement.calculate_disagreement(candidate_instance, base_cf)
        agreement_score = self.calculate_disagreement(candidate_instance, base_cf)  # modified from above line
        return agreement_score

    def calculate_sparsity(self, counterfactual):
        """
        Calculate the sparsity score for a counterfactual.

        @param counterfactual: The counterfactual instance.
        @type counterfactual: list of float

        @return: The sparsity score and the number of changes in the counterfactual.
        @rtype: tuple (float, int)
        """
        num_changes = 0

        for i in range(len(counterfactual)):
            if counterfactual[i] != self.data_instance[i]:
                num_changes += 1

        sparsity = num_changes / len(counterfactual)

        return sparsity, num_changes

    def calculate_entropy(self):
        base_counterfactual_entropies = []
        # Loop for each base counterfactual
        for base_counterfactual in self.base_counterfactuals:
            actions = []

            # Loop for each feature
            for i in range(len(self.data_instance)):
                difference = self.data_instance[i] - base_counterfactual[i]
                if difference < 0:
                    actions.append("INC")
                elif difference > 0:
                    actions.append("DEC")
                else:
                    actions.append("NONE")

            positive_actions = actions.count("INC")
            negative_actions = actions.count("DEC")
            no_actions = actions.count("NONE")
            total_actions = len(actions)

            feature_entropies = []

            # Loop for each feature
            for i in range(len(self.data_instance)):
                positive_entropy = 0
                negative_entropy = 0
                no_entropy = 0

                if (no_actions != 0):
                    no_entropy = (-(no_actions / total_actions) * np.log2(no_actions / total_actions))

                if (positive_actions != 0):
                    positive_entropy = (-(positive_actions / total_actions) * np.log2(positive_actions / total_actions))

                if (negative_actions != 0):
                    negative_entropy = (-(negative_actions / total_actions) * np.log2(negative_actions / total_actions))

                feature_entropy = positive_entropy + negative_entropy + no_entropy
                feature_entropies.append(feature_entropy)

            base_counterfactual_entropy = np.mean(feature_entropies)
            base_counterfactual_entropies.append(base_counterfactual_entropy)
        disagreement = np.mean(base_counterfactual_entropies)

        return disagreement

    def calculate_direction_overlap(self, instance1, instance2):  # Check instance1 not used

        cf_actions = []
        for base_counterfactual in self.base_counterfactuals + instance2:
            actions = []
            for feature in range(len(self.data_instance)):
                difference = self.data_instance[feature] - base_counterfactual[feature]

                if (difference < 0):
                    actions.append("INC")
                elif (difference > 0):
                    actions.append("DEC")
                else:
                    actions.append("NONE")
                cf_actions.append(actions)

        direction_overlap_scores = []

        for i in range(len(self.base_counterfactuals)):
            manhattan_distance = self.calculate_manhattan_distance(self.base_counterfactuals[i], instance2) / len(
                instance2)
            matching_count = sum(
                1 for counterfactual_i_feature, counterfactual_j_feature in zip(cf_actions[i], cf_actions[-1]) if
                counterfactual_i_feature == counterfactual_j_feature)
            direction_overlap = 1 - (matching_count / len(self.data_instance))
            manhattan_direction_overlap = (self.proxmity_weight * manhattan_distance) + (
                        self.overlap_weight * direction_overlap)
            direction_overlap_scores.append(manhattan_direction_overlap)

        average_direction_overlap_score = np.mean(direction_overlap_scores)

        return average_direction_overlap_score

    def validate_instances(self, instance1, instance2):
        """
        Validate that the input instances have the same length and contain only numbers.

        @param instance1: The first instance.
        @type instance1: list of int or float

        @param instance2: The second instance.
        @type instance2: list of int or float

        @raises AssertionError: If input instances have different lengths or contain non-numeric values.
        """
        assert len(instance1) == len(instance2), "Input instances must have the same length."
        assert len(instance1) > 0 and len(instance2) > 0, "Instances must not be empty"

        for element in instance1 + instance2:
            if not isinstance(element, numbers.Number):
                raise AssertionError("Instances must only contain numeric values.")

    def misclassification_penalty(self, counterfactual):
        """
        Calculate the misclassification penalty score for a counterfactual.

        @param counterfactual: The counterfactual instance.
        @type counterfactual: list of float

        @return: The misclassification penalty score.
        @rtype: float
        """
        probability = self.predict_proba_fn(counterfactual)
        return np.dot(probability, self.instance_probability)

    def misclassification_penalty_wachter(self, counterfactual):
        """
        Calculate the Wacther misclassification penalty score for a counterfactual.

        @param counterfactual: The counterfactual instance.
        @type counterfactual: list of float

        @return: The misclassification penalty score.
        @rtype: float
        """
        probability = self.predict_proba_fn(counterfactual)
        probability_target = probability[self.target_class]
        instance_probability_target = self.instance_probability[self.target_class]

        return 1 - abs(probability_target - instance_probability_target)