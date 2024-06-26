# This is a fork of auto-sklearn repository, not the original auto-sklearn!
I have added a functionality to the pre-processor to transform labels. This allows the pre-processor to add/remove instances, or fix noise in the labels.

# Details
I have replaced Pipeline from sklearn to Pipeline of imblearn. This allowed to add SMOTE-like to balancing step of auto-sklearn.

# Learn more
- Learn more about [auto-sklearn](https://automl.github.io/auto-sklearn/master/).
- Learn more about [sklearn](https://automl.github.io/auto-sklearn/master/).
- Learn more about [imblearn](https://imbalanced-learn.org/stable/).
