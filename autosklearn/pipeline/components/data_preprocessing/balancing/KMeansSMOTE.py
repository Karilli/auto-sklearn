from typing import Dict, Optional, Tuple, Union
import imblearn.over_sampling as imblearn
from sklearn.cluster import MiniBatchKMeans
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



# TODO: This error message appears a lot: RuntimeError: No clusters found with sufficient 
# samples of class 0.0. Try lowering the cluster_balance_threshold or increasing the number of clusters.

class KMeansSMOTE(AutoSklearnPreprocessingAlgorithm):
    def __init__(
            self, 
            sampling_strategy=1.0, 
            k_neighbors=2,
            cluster_balance_threshold=1.0,
            density_exponent=1.0,

            n_clusters=8,
            init="k-means++",

            random_state=None
        ) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.n_clusters = n_clusters
        self.init = init

        self.random_state = random_state

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        return imblearn.KMeansSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
            kmeans_estimator=MiniBatchKMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                random_state=self.random_state
            )
        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "KMeansSMOTE",
            "name": "KMeansSMOTE",
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
            UniformIntegerHyperparameter("k_neighbors", 2, 10, default_value=2),
            UniformFloatHyperparameter("cluster_balance_threshold", 0.0, 1.0, default_value=dataset_properties["imbalanced_ratio"], log=False), 
            UniformFloatHyperparameter("density_exponent", 1.0, 2.0, default_value=1.0, log=False), 

            UniformIntegerHyperparameter("n_clusters", 10, 30, default_value=20),
            CategoricalHyperparameter("init", ["k-means++", "random"], "k-means++"),
        ])
        return cs
