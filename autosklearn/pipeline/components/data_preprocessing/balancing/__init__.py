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
    def __init__(self, strategy="none", random_state=None):
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

        # TODO check for task type classification and/or regression!

        components_dict = OrderedDict()
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
            defaults = ["no_preprocessing", "SMOTETomek", "SMOTEENN", "SVMSMOTE", "EditedNearestNeighbours"]
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

    def fit_resample(self, X, y):
        return self.choice.fit_resample(X, y)

    @staticmethod
    def get_weights(
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

    def __repr__(self):
        return f"BalancingChoice(weighting={self.strategy == 'weighting'}, resampling={self.choice})" 
