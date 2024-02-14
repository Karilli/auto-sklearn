from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import PIPELINE_DATA_DTYPE

from typing import Optional, Type

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.askl_typing import FEAT_TYPE_TYPE


from ...base import (
    AutoSklearnChoice,
    AutoSklearnPreprocessingAlgorithm,
    ThirdPartyComponents,
    _addons,
    find_components,
)

balancing_directory = os.path.split(__file__)[0]
_preprocessors = find_components(
    __package__, balancing_directory, AutoSklearnPreprocessingAlgorithm
)

additional_components = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)
_addons["data_preprocessing.balancing"] = additional_components


def add_preprocessor(preprocessor: Type[AutoSklearnPreprocessingAlgorithm]) -> None:
    additional_components.add_component(preprocessor)


class BalancingChoice(AutoSklearnChoice):
    # TODO: NOTE: This is a disgusting hack, !it will probably break in multithreaded context!
    # We have to create weights for classifiers and feature_preprocessor before we run Pipeline.fit()
    # in order to pass them as fit_params. But if we create weights before we run the Pipeline, methods 
    # from imblearn will invalidate them, by tempering with labels. 
    # The solution is to pass a pointer which can be later modified during the Pipeline evaluation.
    # Alternative solution is to have only single step Balancing, i.e. either weighting, or resampling. 
    # Instead of two step Balancing, i.e. resampling and then also weighting.
    SAMPLE_WEIGHTS=None
    CLASS_WEIGHTS=None
    def __init__(self, strategy="none", feat_type=None, dataset_properties=None, random_state=None):
        self.strategy = strategy
        self.random_state = random_state

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(additional_components.components)
        return components

    def get_available_components(
        self, dataset_properties=None, include=None, exclude=None
    ):
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together."
            )

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError(
                        "Trying to include unknown component: " "%s" % incl
                    )

        components_dict = OrderedDict()
        # TODO: Maybe prevent some or all balancing methods to get into hyper_parameter_space
        # if the dataset_properties["imbalanced_ratio"] is relatively high?

        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == BalancingChoice or hasattr(entry, "get_components"):
                continue

            target_type = dataset_properties["target_type"]
            if target_type == "classification":
                if entry.get_properties()["handles_classification"] is False:
                    continue
                if (
                    dataset_properties.get("multiclass") is True
                    and entry.get_properties()["handles_multiclass"] is False
                ):
                    continue
                if (
                    dataset_properties.get("multilabel") is True
                    and entry.get_properties()["handles_multilabel"] is False
                ):
                    continue

            elif target_type == "regression":
                raise ValueError("Balancing is not allowed for regression tasks.")

            else:
                raise ValueError("Unknown target type %s" % target_type)

            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties=None,
        default=None,
        include=None,
        exclude=None,
    ):
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
            raise ValueError("No preprocessors found, please add no_preprocessing")

        if default is None:
            defaults = ["no_preprocessing", "SVMSMOTE", "RepeatedEditedNearestNeighbours", "SMOTEENN", "SMOTETomek", "BorderlineSMOTE", "SMOTE", "KMeansSMOTE", "TomekLinks"]
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter(
            "__choice__", list(available_preprocessors.keys()), default_value=default
        )
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[
                name
            ].get_hyperparameter_search_space(dataset_properties=dataset_properties)
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )

        # TODO: 'strategy' is not an ideal name, better would be 'weighting' with
        # options True/False. But for the reason to keep META-learninig database
        # compatible it is named like this.
        # TODO: This parameter doesn't have corresponding 'get_properties' method,
        # there probably should be some check if this parameter should be included?
        # TODO: possible optimization is to make another parameter 'sampling_strategy'
        # and share one global parameter between all of the resamplers
        cs.add_hyperparameter(
            CategoricalHyperparameter("strategy", ["none", "weighting"], "none")
        )
        return cs
    
    def set_hyperparameters(
        self,
        configuration,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params=None,
    ):
        new_params = {}

        params = configuration.get_dictionary()
        choice = params["__choice__"]
        del params["__choice__"]

        for param, value in params.items():
            if param == "strategy":
                self.strategy = value
            else:
                param = param.replace(choice, "").replace(":", "")
                new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, "").replace(":", "")
                new_params[param] = value

        new_params["random_state"] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def fit_resample(self, X, y):
        try:
            X, y = self.choice.fit_resample(X, y)
        # TODO: How to set up sample_strategy in configuration spaces of resamplers to completely avoid these errors?
        # TODO: How to set up KMeansSMOTE n_clusters and cluster_balance_threshold parameters?
        except ValueError as e:
            if str(e) == 'The specified ratio required to remove samples from the minority class while trying to generate new samples. Please increase the ratio.' or \
                str(e) == 'No samples will be generated with the provided ratio settings.':
                _, (c1, c2) = np.unique(y, return_counts=True)
                (c1, c2) = sorted((c1, c2))
                raise ValueError(f"Error with sample_strategy: ratio={c1 / c2}, sampling_strategy={self.choice.sampling_strategy}")
            raise e
        except RuntimeError as e:
            if str(e) == 'No clusters found with sufficient samples of class 0.0. Try lowering the cluster_balance_threshold or increasing the number of clusters.':
                raise RuntimeError("Error with KMeansSMOTE n_clusters and cluster_balance_threshold")
        self.set_weights(y)
        return X, y

    @classmethod
    def prepare_params_for_weighting(
        cls,
        classifier: BaseEstimator,
        preprocessor: BaseEstimator,
        init_params: Dict[str, Any],
        fit_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
            cls.SAMPLE_WEIGHTS = np.array([])
            if classifier in clf_:
                fit_params["classifier:sample_weight"] = cls.SAMPLE_WEIGHTS
            if preprocessor in pre_:
                fit_params["feature_preprocessor:sample_weight"] = cls.SAMPLE_WEIGHTS
        else:
            cls.SAMPLE_WEIGHTS = None

        clf_ = ["decision_tree", "liblinear_svc", "libsvm_svc"]
        pre_ = ["liblinear_svc_preprocessor", "extra_trees_preproc_for_classification"]
        if classifier in clf_:
            init_params["classifier:class_weight"] = "balanced"
        if preprocessor in pre_:
            init_params["feature_preprocessor:class_weight"] = "balanced"

        clf_ = ["ridge"]
        if classifier in clf_:
            cls.CLASS_WEIGHTS = {}
            init_params["classifier:class_weight"] = cls.CLASS_WEIGHTS
        else:
            cls.CLASS_WEIGHTS = None

        return init_params, fit_params

    def set_weights(self, Y: PIPELINE_DATA_DTYPE):
        if self.SAMPLE_WEIGHTS is not None:
            if len(Y.shape) > 1:
                offsets = [2**i for i in range(Y.shape[1])]
                Y_ = np.sum(Y * offsets, axis=1)
            else:
                Y_ = Y

            self.SAMPLE_WEIGHTS.resize(Y_.shape, refcheck=False)
            self.SAMPLE_WEIGHTS.fill(1)

            unique, counts = np.unique(Y_, return_counts=True)
            # This will result in an average weight of 1!
            cw = 1 / (counts / np.sum(counts)) / 2
            if len(Y.shape) == 2:
                cw /= Y.shape[1]

            for i, ue in enumerate(unique):
                mask = Y_ == ue
                self.SAMPLE_WEIGHTS[mask] *= cw[i]

        if self.CLASS_WEIGHTS is not None:
            unique, counts = np.unique(Y, return_counts=True)
            cw = 1.0 / counts
            cw = cw / np.mean(cw)

            for i, ue in enumerate(unique):
                self.CLASS_WEIGHTS[ue] = cw[i]

    def __repr__(self):
        return f"BalancingChoice(weighting={self.strategy == 'weighting'}, resampling={repr(self.choice)})" 
