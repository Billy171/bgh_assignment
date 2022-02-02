from typing import Dict, List, Union

import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def create_data_splits(
    df: pd.DataFrame, test_size_absolute: int
) -> Dict[str, pd.DataFrame]:
    """Slices input data into training and test splits. Conventional terminology adopted."""

    df_test = df.sample(n=test_size_absolute, replace=False, random_state=0)
    test_indecies = df_test.index
    df_train = df.drop(test_indecies)
    return dict(df_train=df_train, df_test=df_test)


def train_model(
    df_train: pd.DataFrame, target_column_name: str, ignored_columns: List[str]
) -> Union[RegressorMixin, ClassifierMixin]:
    """Selects and trains an appropriate supervised learning model with the supplied input data."""

    #dropping ignored cols
    df_train = df_train.drop(labels=ignored_columns, axis=1)

    #removing rows with null values (only losing ~250 of 5593 here)
    df_train = df_train.dropna()
    X = df_train.drop(target_column_name, axis=1)
    y = df_train[target_column_name]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


def test_model(
    df_test: pd.DataFrame, model: Union[RegressorMixin, ClassifierMixin],
    target_column_name: str, ignored_columns: List[str]
) -> pd.DataFrame:
    """Tests the model on a held out test set"""

    # dropping ignored cols
    df_test = df_test.drop(labels=ignored_columns, axis=1)

    #removing rows with null values (on;y losing ~200 of 5000 here)
    df_test = df_test.dropna()
    X = df_test.drop(target_column_name, axis=1)
    y = df_test[target_column_name]
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    return mse


if __name__ == "__main__":
    df = pd.read_csv("sample_data_format.csv")
    train_test_dict = create_data_splits(df=df, test_size_absolute=5000)
    df_train, df_test = train_test_dict["df_train"], train_test_dict["df_test"]
    model = train_model(df_train, target_column_name="R_T1W", ignored_columns=["Date", "Coin"])
    results = test_model(df_test=df_test, model=model, target_column_name="R_T1W",
                         ignored_columns=["Date", "Coin"])
    print(results)






