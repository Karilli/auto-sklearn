from typing import Any, Dict, Optional, Tuple

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

# NOTE: all of the "AutoSklearnPreprocessingAlgorithm"s must be imported after call to "find_components"
# find_components relies on order of imports. The function doesn't add components from previously imported
# packages.
from .no_balancing import NoBalancing  # noqa
from .weighting import Weighting  # noqa
from .SMOTE import SMOTE  # noqa


def add_preprocessor(preprocessor: Type[AutoSklearnPreprocessingAlgorithm]) -> None:
    additional_components.add_component(preprocessor)


class BalancingChoice(AutoSklearnChoice):
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
            raise ValueError("No preprocessors found, please add no_balancing")

        if default is None:
            defaults = ["no_balancing", "weighting"]
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

        return cs

    def transform(self, X):
        return self.choice.transform(X)
