from typing import Any, Dict, List, Optional, Tuple, Union
import imblearn.over_sampling as imblearn
from sklearn.svm import SVC
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

class SVMSMOTE(AutoSklearnPreprocessingAlgorithm):
    def __init__(
            self, 
            sampling_strategy=1.0, 
            k_neighbors=5, 
            m_neighbors=10,
            out_step=0.5,

            C=1.0,
            kernel="rbf",
            degree=3,
            shrinking=True,
            random_state=None
        ) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.out_step = out_step

        self.C=C
        self.kernel = kernel
        self.degree = degree
        self.shrinking = shrinking

        self.random_state = random_state

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        return imblearn.SVMSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors
            out_step=self.out_step,
            random_state=self.random_state,
            svm_estimator=SVC(
                C=self.C,
                kernel=self.kernel,
                degree=self.degree,
                shrinking=self.shrinking,
                random_state=self.random_state
            )
        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "SMOTE",
            "name": "SMOTE",
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
            UniformFloatHyperparameter("sampling_strategy", 0.0, 1.0, default_value=1.0, log=False), 
            UniformIntegerHyperparameter("k_neighbors", 1, 20, default_value=5),
            UniformIntegerHyperparameter("m_neighbors", 1, 50, default_value=10),
            UniformFloatHyperparameter("out_step", 0.0, 1.0, default_value=0.5, log=False), 
            
            UniformFloatHyperparameter("C", 1.0, 1000.0, default_value=1.0, log=True), 
            CategoricalHyperparameter("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"], "rbf"),
            UniformIntegerHyperparameter("degree", 1, 10, default_value=3),
            CategoricalHyperparameter("shrinking", [True, False], True)
        ])
        return cs
