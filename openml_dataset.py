import sklearn.model_selection
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import openml as oml
import sys

import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
import numpy as np #added

def get_openml_classification_dataset(id):
    dataset = oml.datasets.get_dataset(id)

    X, y, cat_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    ydf = pd.DataFrame(y)
    rows = len(X)
    classes = len(ydf.iloc[:,0].value_counts())
    columns = np.append(attribute_names, dataset.default_target_attribute) #added by Maya
    frequencies = ydf.iloc[:,0].value_counts()
    dummy = frequencies.iat[0]/len(X)

    ### Label Encoding
    encoder = preprocessing.LabelEncoder()
    try:
        encoder.fit(y)
        y = encoder.transform(y)
    except:
        print(sys.exc_info()[0])

    imputer = SimpleImputer(strategy='most_frequent')
    values = imputer.fit_transform(X.values)

    cat_count = 0
    for i in range(len(attribute_names)):
        if cat_indicator[i] == True:
            try:
                v = values[:,i]
                train = encoder.fit_transform(v)
                values[:,i] = train
                cat_count = cat_count + 1
            except:
                print(sys.exc_info()[0])
                continue

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(values, y, random_state=1)
    X_train = np.array(list(X_train), dtype=float)
    X_test = np.array(list(X_test), dtype=float)

    return (X_train, X_test, y_train, y_test, dataset.name, dummy, rows, classes, columns) #changed by Maya, orig: X.columns)
