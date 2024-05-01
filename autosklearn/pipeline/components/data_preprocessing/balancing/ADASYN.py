from typing import Dict, Optional, Tuple, Union

import imblearn.over_sampling as imblearn
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

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


class ADASYN(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, sampling_strategy=1.0, n_neighbors=5, random_state=None) -> None:
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        return imblearn.ADASYN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "ADASYN",
            "name": "ADASYN",
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
        assert "imbalanced_ratio" in dataset_properties, dataset_properties

        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "sampling_strategy",
                    min(0.99999, dataset_properties["imbalanced_ratio"] + 0.01),
                    1.0,
                    default_value=1.0,
                    log=False,
                ),
                UniformIntegerHyperparameter("n_neighbors", 3, 10, default_value=5),
            ]
        )
        return cs
