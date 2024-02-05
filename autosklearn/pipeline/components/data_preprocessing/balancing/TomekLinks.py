from typing import Dict, Optional, Tuple, Union
import imblearn.under_sampling as imblearn
from ConfigSpace.configuration_space import ConfigurationSpace
import numpy as np

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

from ConfigSpace.hyperparameters import UniformFloatHyperparameter


# TODO: add sampling_strategy parameter
class TomekLinks(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, sampling_strategy=1.0, random_state=None) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        (l1, l2), (c1, c2) = np.unique(y, return_counts=True)
        (c1, l1), (c2, l2) = sorted(((c1, l1), (c2, l2)))
        return imblearn.TomekLinks(
            sampling_strategy=lambda: {l1: c1, l2: min(c2, int(c1 / self.sampling_strategy))},
        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "TomekLinks",
            "name": "TomekLinks",
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
            UniformFloatHyperparameter("sampling_strategy", dataset_properties["imbalanced_ratio"] + 0.01, 1.0, default_value=1.0, log=False), 
        ])
        return cs
