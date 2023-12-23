from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import (
    DENSE,
    INPUT,
    SIGNED_DATA,
    SPARSE,
    UNSIGNED_DATA,
)


ENABLE_SMOTE = 1
ENABLE_PARAMS = 0

class Balancing(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        strategy: str = "none",
        sampling_strategy: float = 1.0,
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        out_step: float = 0.5,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.strategy = strategy
        self.random_state = random_state
        self.sampling_strategy= sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.out_step = out_step

    def fit(self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None) -> "Balancing":
        self.fitted_ = True
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE, y) -> PIPELINE_DATA_DTYPE:
        from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, SMOTENC
        from collections import Counter

        if self.strategy == "SMOTE":
            if self.sampling_strategy * len(y) <= min(Counter(y).values()) + 1:
                return X, y
            return SVMSMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                m_neighbors=self.m_neighbors,
                out_step=self.out_step,
                random_state=self.random_state
            ).fit_resample(X, y)
        return X, y

    def get_weights(
        self,
        Y: PIPELINE_DATA_DTYPE,
        classifier: BaseEstimator,
        preprocessor: BaseEstimator,
        init_params: Optional[Dict[str, Any]],
        fit_params: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if init_params is None:
            init_params = {}

        if fit_params is None:
            fit_params = {}

        # Classifiers which require sample weights:
        # We can have adaboost in here, because in the fit method,
        # the sample weights are normalized:
        # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/ensemble/weight_boosting.py#L121
        # Have RF and ET in here because they emit a warning if class_weights
        #  are used together with warmstarts
        clf_ = [
            "adaboost",
            "random_forest",
            "extra_trees",
            "sgd",
            "passive_aggressive",
            "gradient_boosting",
        ]
        pre_: List[str] = []
        if classifier in clf_ or preprocessor in pre_:
            if len(Y.shape) > 1:
                offsets = [2**i for i in range(Y.shape[1])]
                Y_ = np.sum(Y * offsets, axis=1)
            else:
                Y_ = Y

            unique, counts = np.unique(Y_, return_counts=True)
            # This will result in an average weight of 1!
            cw = 1 / (counts / np.sum(counts)) / 2
            if len(Y.shape) == 2:
                cw /= Y.shape[1]

            sample_weights = np.ones(Y_.shape)

            for i, ue in enumerate(unique):
                mask = Y_ == ue
                sample_weights[mask] *= cw[i]

            if classifier in clf_:
                fit_params["classifier:sample_weight"] = sample_weights
            if preprocessor in pre_:
                fit_params["feature_preprocessor:sample_weight"] = sample_weights

        # Classifiers which can adjust sample weights themselves via the
        # argument `class_weight`
        clf_ = ["decision_tree", "liblinear_svc", "libsvm_svc"]
        pre_ = ["liblinear_svc_preprocessor", "extra_trees_preproc_for_classification"]
        if classifier in clf_:
            init_params["classifier:class_weight"] = "balanced"
        if preprocessor in pre_:
            init_params["feature_preprocessor:class_weight"] = "balanced"

        clf_ = ["ridge"]
        if classifier in clf_:
            class_weights = {}

            unique, counts = np.unique(Y, return_counts=True)
            cw = 1.0 / counts
            cw = cw / np.mean(cw)

            for i, ue in enumerate(unique):
                class_weights[ue] = cw[i]

            if classifier in clf_:
                init_params["classifier:class_weight"] = class_weights

        return init_params, fit_params

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "Balancing",
            "name": "Balancing Imbalanced Class Distributions",
            "handles_missing_values": True,
            "handles_nominal_values": True,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        
        cs = ConfigurationSpace()
        strategy = CategoricalHyperparameter("strategy", ["none", "weighting"] + (["SMOTE"] if ENABLE_SMOTE else []), default_value="none")
        cs.add_hyperparameter(strategy)

        if ENABLE_SMOTE and ENABLE_PARAMS:
            cs_SMOTE = ConfigurationSpace()
            sampling_strategy = UniformFloatHyperparameter(name="sampling_strategy", lower=0.0, upper=1.0, default_value=1.0, log=False)
            k_neighbors = UniformIntegerHyperparameter(name="k_neighbors", lower=1, upper=10, default_value=5, log=False)
            m_neighbors = UniformIntegerHyperparameter(name="m_neighbors", lower=1, upper=20, default_value=10, log=False)
            out_step = UniformFloatHyperparameter(name="out_step", lower=0.0, upper=1.0, default_value=0.5, log=False)
            cs_SMOTE.add_hyperparameters([sampling_strategy, k_neighbors, m_neighbors, out_step])
            cs.add_configuration_space(
                "SMOTE",
                cs_SMOTE,
                parent_hyperparameter={"parent": strategy, "value": "SMOTE"}
            )
        return cs
