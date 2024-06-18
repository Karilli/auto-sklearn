# Create new project and virtual environment
```bash
mkdir new-project
cd new-project
python3 -m venv auto-sklearn-venv
source auto-sklearn-venv/bin/activate
```

# Install auto-sklearn with SMOTE
```bash
git clone --recurse-submodules https://github.com/Karilli/auto-sklearn.git
cd auto-sklearn
git checkout SMOTE-version-6
pip install -e ".[test,doc,examples]"
```

# Create a python script that uses auto-sklearn with SMOTE
# 
```bash
cd ..
touch example.py
```

# Copy-paste the following code into example.py

```Python3
import sklearn.datasets

import sys
# modify this to path corresponding to ./auto-sklearn/autosklearn
sys.path.insert(0, "./path/to/auto-sklearn/autosklearn")
from autosklearn import AutoSklearnClassifier

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

automl = AutoSklearnClassifier(
    time_left_for_this_task=120,
    include={"balancing": ["SVMSMOTE"]},
    per_run_time_limit=30
).fit(X_train, y_train, dataset_name="breast_cancer")

print(automl.show_models())
```

# Run the script, you should see SVMSMOTE in all balancing steps
```bash
python3 example.py
```


