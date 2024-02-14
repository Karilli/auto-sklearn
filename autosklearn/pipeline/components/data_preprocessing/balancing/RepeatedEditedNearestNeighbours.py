import numpy as np
from typing import Dict, Optional, Tuple, Union
import imblearn.under_sampling as imblearn
from ConfigSpace.configuration_space import ConfigurationSpace

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

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)


class RepeatedEditedNearestNeighbours(AutoSklearnPreprocessingAlgorithm):
    def __init__(
            self, 
            sampling_strategy="not minority",
            n_neighbors=5,
            kind_sel="all",
            max_iter=100,
            random_state=None
        ) -> None:
        self.sampling_strategy = sampling_strategy
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.random_state = random_state

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        return imblearn.RepeatedEditedNearestNeighbours(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            max_iter=self.max_iter,
        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "RepeatedEditedNearestNeighbours",
            "name": "RepeatedEditedNearestNeighbours",
            "handles_missing_values": False,
            "handles_nominal_values": False,
            "handles_numerical_features": True,
            "prefers_data_scaled": True,
            "prefers_data_normalized": True,
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            # TODO: I dont know how to set the following 6 properties.
            "is_deterministic": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            CategoricalHyperparameter("sampling_strategy", ["not minority", "majority", "all"], "not minority"),
            UniformIntegerHyperparameter("n_neighbors", 3, 10, default_value=3),
            UniformIntegerHyperparameter("max_iter", 1, 1000, default_value=100, log=True),
            CategoricalHyperparameter("kind_sel", ["all", "mode"], "all"),
        ])
        return cs
