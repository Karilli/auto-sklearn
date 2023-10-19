from sklearn.pipeline import Pipeline, _final_estimator_has

from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if

from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import Balancing


def _fit_transform_one_fixed(
    transformer, X, y, weight, message_clsname="", message=None, **fit_params
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """

    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            if isinstance(transformer, Balancing):
                Xt, yt = transformer.fit_transform(X, y, **fit_params)
            else:
                Xt = transformer.fit_transform(X, y, **fit_params)
        else:
            if isinstance(transformer, Balancing):
                Xt, yt = transformer.fit(X, y, **fit_params).transform(X, y)
            else:
                Xt = transformer.fit(X, y, **fit_params).transform(X)

    if isinstance(transformer, Balancing):
        if weight is None:
            return Xt, yt, transformer
        return Xt * weight, yt, transformer
    if weight is None:
        return Xt, transformer
    return Xt * weight, transformer


def _fit_fixed(self, X, y=None, **fit_params_steps):
    # shallow copy of steps - this should really be steps_
    self.steps = list(self.steps)
    self._validate_steps()
    # Setup the memory
    memory = check_memory(self.memory)

    fit_transform_one_cached = memory.cache(_fit_transform_one_fixed)

    encountered_balancing = False
    for step_idx, name, transformer in self._iter(with_final=False, filter_passthrough=False):
        if transformer is None or transformer == "passthrough":
            with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                continue

        if hasattr(memory, "location") and memory.location is None:
            # we do not clone when caching is disabled to
            # preserve backward compatibility
            cloned_transformer = transformer
        else:
            cloned_transformer = clone(transformer)

        if isinstance(cloned_transformer, Balancing):
            encountered_balancing = True
            X, y, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
        else:
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
        # Replace the transformer of the step with the fitted
        # transformer. This is necessary when loading the transformer
        # from the cache.
        self.steps[step_idx] = (name, fitted_transformer)

    if encountered_balancing:
        return X, y
    return X


@available_if(_final_estimator_has("predict_proba"))
def predict_proba_fixed(self, X, **predict_proba_params):
    """Transform the data, and apply `predict_proba` with the final estimator.

    Call `transform` of each transformer in the pipeline. The transformed
    data are finally passed to the final estimator that calls
    `predict_proba` method. Only valid if the final estimator implements
    `predict_proba`.

    Parameters
    ----------
    X : iterable
        Data to predict on. Must fulfill input requirements of first step
        of the pipeline.

    **predict_proba_params : dict of string -> object
        Parameters to the `predict_proba` called at the end of all
        transformations in the pipeline.

    Returns
    -------
    y_proba : ndarray of shape (n_samples, n_classes)
        Result of calling `predict_proba` on the final estimator.
    """
    Xt = X
    for _, name, transform in self._iter(with_final=False):
        if isinstance(transform, Balancing) and (transform.strategy in ("none", "weighting")):
            Xt, _ = transform.transform(Xt, None)
        elif not isinstance(transform, Balancing):
            Xt = transform.transform(Xt)
    return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)


Pipeline._fit = _fit_fixed
Pipeline.predict_proba = predict_proba_fixed
