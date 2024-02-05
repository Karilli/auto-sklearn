from typing import Dict, Optional, Tuple, Union
import imblearn.combine as imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
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
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)


class SMOTETomek(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self, 
        smote_sampling_strategy=1.0, 
        smote_k_neighbors=5,
        tomek_sampling_strategy="not minority",

        random_state=None
    ) -> None:
        self.smote_sampling_strategy = smote_sampling_strategy
        self.tomek_sampling_strategy = tomek_sampling_strategy
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        return imblearn.SMOTETomek(
            smote=SMOTE(
                sampling_strategy=self.smote_sampling_strategy,
                k_neighbors=self.smote_k_neighbors,
                random_state=self.random_state
            ),
            tomek=TomekLinks(
                sampling_strategy=self.tomek_sampling_strategy
            ),
            random_state=self.random_state
        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "SMOTETomek",
            "name": "SMOTETomek",
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
            UniformFloatHyperparameter("smote_sampling_strategy", dataset_properties["imbalanced_ratio"] + 0.01, 1.0, default_value=1.0, log=False),  
            UniformIntegerHyperparameter("smote_k_neighbors", 3, 10, default_value=5),
            CategoricalHyperparameter("tomek_sampling_strategy", ["not minority", "majority", "all"], "not minority"),
        ])
        # TODO: "ValueError('Expected n_neighbors <= n_samples,  but n_samples = 8, n_neighbors = 11')"
        return cs
