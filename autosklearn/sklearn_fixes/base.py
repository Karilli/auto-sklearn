from sklearn.base import TransformerMixin


def fit_transform_fixed(self, X, y=None, **fit_params):
    """
    Fit to data, then transform it.

    Fits transformer to `X` and `y` with optional parameters `fit_params`
    and returns a transformed version of `X`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input samples.

    y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        Target values (None for unsupervised transformations).

    **fit_params : dict
        Additional fit parameters.

    Returns
    -------
    X_new : ndarray array of shape (n_samples, n_features_new)
        Transformed array.
    """
    from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import (
        Balancing,
    )

    # non-optimized default implementation; override when a better
    # method is possible for a given clustering algorithm

    if isinstance(self, Balancing):
        return self.fit(X, y, **fit_params).transform(X, y)
    elif y is None:
        # fit method of arity 1 (unsupervised transformation)
        return self.fit(X, **fit_params).transform(X)
    else:
        # fit method of arity 2 (supervised transformation)
        return self.fit(X, y, **fit_params).transform(X)


TransformerMixin.fit_transform = fit_transform_fixed
