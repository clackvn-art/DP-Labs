import pandas as pd
import numpy as np
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Lab_8.csv.csv")

# BÀI 1
#chia cột
num_cols = ["LotArea", "Rooms", "NoiseFeature"]
cat_cols = ["HasGarage", "Neighborhood", "Condition"]
text_col = "Description"
time_col = ["SaleDate"]

class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        d = pd.to_datetime(X.iloc[:,0])

        return pd.DataFrame({
            "year": d.dt.year,
            "month": d.dt.month,
            "quarter": d.dt.quarter
        })

#  ColumnTransformer
num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

text_pipe = Pipeline([
("tfidf", TfidfVectorizer(stop_words="english", max_features=20))
])

time_pipe = Pipeline([
    ('date', DateFeatures())
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
    ("text", text_pipe, text_col),
    ("time", time_pipe, time_col)
])

def B1():
    X = df.drop("SalePrice", axis=1)
    Xt = preprocessor.fit_transform(X)
    print("Shape:", Xt.shape)
# B1()

def B2():
    tests = {
        "full": df.copy(),

        "missing":
            df.copy().mask(np.random.random(df.shape) < 0.1),

        "skewed":
            df.assign(LotArea=df["LotArea"] * 50),

        "unseen":
            df.assign(Neighborhood="NewTown"),

        "wrong_type":
            df.assign(LotArea="abc")
    }

    for name, data in tests.items():
        try:
            Xt = preprocessor.fit_transform(data.drop("SalePrice", axis=1))
            print(name, "OK", Xt.shape)
        except Exception as e:
            print(name, "ERROR:", e)
# B2()

def B3():
    warnings.filterwarnings("ignore")
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            cv=5,
            scoring="r2",
            return_train_score=False
        )

        print(name)
        print("R2:", scores["test_score"].mean())
# B3()

