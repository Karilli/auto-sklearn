from typing import Dict, Optional, Tuple, Union
import imblearn.under_sampling as imblearn
from sklearn.cluster import KMeans
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
    CategoricalHyperparameter,
    UniformIntegerHyperparameter
)

class ClusterCentroids(AutoSklearnPreprocessingAlgorithm):
    def __init__(
            self, 
            sampling_strategy=1.0, 
            voting="soft",

            n_clusters=8,
            init="k-means++",
            n_init=1,
            algorithm="lloyd",
            tol=1e-4,

            random_state=None
        ) -> None:
        self.sampling_strategy = sampling_strategy
        self.voting = voting

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.algorithm = algorithm
        self.tol = tol

        self.random_state = random_state

    def fit_resample(
        self, X: PIPELINE_DATA_DTYPE, y: PIPELINE_DATA_DTYPE
    ) -> Tuple[PIPELINE_DATA_DTYPE, PIPELINE_DATA_DTYPE]:
        return imblearn.ClusterCentroids(
            sampling_strategy=self.sampling_strategy,
            voting=self.voting,
            estimator=KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                algorithm=self.algorithm,
                tol=self.tol,
                random_state=self.random_state
            ),
            random_state=self.random_state

        ).fit_resample(X, y)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "ClusterCentroids",
            "name": "ClusterCentroids",
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
            UniformFloatHyperparameter("sampling_strategy", 0.0, 1.0, 1.0, log=False), 
            CategoricalHyperparameter("voting", ["hard", "soft"], "soft"),

            UniformIntegerHyperparameter("n_clusters", 2, 20, default_value=8),
            CategoricalHyperparameter("init", ["k-means++", "random"], "k-means++"),
            UniformIntegerHyperparameter("n_init", 1, 20, 1),
            CategoricalHyperparameter("algorithm", ["lloyd", "elkan"], "lloyd"),
            UniformFloatHyperparameter("tol", 1e-5, 1e-1, 1e-4, log=True),
        ])
        return cs
