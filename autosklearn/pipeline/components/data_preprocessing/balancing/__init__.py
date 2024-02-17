from typing import Any, Dict, List, Optional, Tuple, Type

import os
from collections import OrderedDict

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import PIPELINE_DATA_DTYPE

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
    def __init__(
        self,
        strategy="none",
        feat_type=None,
        dataset_properties=None,
        random_state=None,
    ):
        self.strategy = strategy
        self.random_state = random_state

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(additional_components.components)
        for new, old in ("none", "no_preprocessing"), ("weighting", "Weighting"):
            if old in components:
                components[new] = components[old]
                del components[old]
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
        # TODO: Maybe prevent some or all balancing methods to get into
        # hyper_parameter_space if the dataset_properties["imbalanced_ratio"]
        # is relatively high?

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

        # Compile a list of legal resamplers for this problemm, 0 is allowed
        available_components = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if default is None:
            defaults = [
                "none",
                "weighting",
                "SVMSMOTE",
                "BorderlineSMOTE",
                "RepeatedEditedNearestNeighbours",
                "SMOTEENN",
                "SMOTETomek",
            ] + list(available_components)
            for default_ in defaults:
                if default_ in available_components:
                    default = default_
                    break

        strategy = CategoricalHyperparameter(
            "strategy", list(available_components.keys()), default_value=default
        )
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)

        for name in available_components:
            preprocessor_configuration_space = available_components[
                name
            ].get_hyperparameter_search_space(dataset_properties=dataset_properties)
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter={"parent": strategy, "value": name},
            )

        # TODO: possible optimization is to make 'global' parameter 'sampling_strategy'
        # and share it between all of the resamplers
        return cs

    def set_hyperparameters(
        self,
        configuration,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params=None,
    ):
        params = configuration.get_dictionary()
        choice = params["strategy"]
        del params["strategy"]

        self.new_params = {}

        for param, value in params.items():
            param = param.replace(choice, "").replace(":", "")
            self.new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, "").replace(":", "")
                self.new_params[param] = value

        self.new_params["random_state"] = self.random_state
        self.choice = self.get_components()[choice](**self.new_params)

        return self

    def fit_resample(self, X, y):
        try:
            X, y = self.choice.fit_resample(X, y)
        # TODO: How to set up sample_strategy in configuration spaces of resamplers
        # to completely avoid these errors?
        # TODO: How to set up KMeansSMOTE n_clusters and cluster_balance_threshold
        # parameters?
        except ValueError as e:
            if (
                str(e)
                == "The specified ratio required to remove samples from the minority "
                "class while trying to generate new samples. Please increase the ratio."
                or str(e)
                == "No samples will be generated with the provided ratio settings."
            ):
                _, (c1, c2) = np.unique(y, return_counts=True)
                (c1, c2) = sorted((c1, c2))
                raise ValueError(
                    f"Error with sample_strategy: ratio={c1 / c2}, "
                    f"sampling_strategy={self.choice.sampling_strategy}"
                )
            raise e
        except RuntimeError as e:
            if (
                str(e) == "No clusters found with sufficient samples of class 0.0. Try "
                "lowering the cluster_balance_threshold or increasing the number "
                "of clusters."
            ):
                raise RuntimeError(
                    "Error with KMeansSMOTE n_clusters and cluster_balance_threshold"
                )
        return X, y

    @classmethod
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
