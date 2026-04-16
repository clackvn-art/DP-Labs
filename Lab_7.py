import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Lab_7.csv")

def B1():
    #top 10 cột lệch nhất. 
    num_cols = df.select_dtypes(include=np.number).columns
    print(num_cols)
    skew_table = df[num_cols].skew().sort_values(ascending=False)
    print(skew_table.head(10))

    # biểu đồ
    top3 = skew_table.head(3).index
    plt.figure(figsize=(15,4))

    for i, col in enumerate(top3, 1):
        plt.subplot(1,3,i)
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"{col} | Skew={df[col].skew():.2f}")

    plt.tight_layout()
    plt.show()

    #outlier 
    for col in top3:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

        print(outliers)

    #phương pháp biến đổi (log)
    for col in top3:
        df[col + "_log"] = np.log1p(df[col])
        sns.histplot(df[col + "_log"], kde=True)
        plt.show()
# B1()

def B2():
    num_cols = df.select_dtypes(include=np.number).columns
    print(num_cols)
    skew_table = df[num_cols].skew().sort_values(ascending=False)
    print(skew_table)

    # chọn cột
    col1 = "SalePrice"
    col2 = "LotArea"
    col3 = "NegSkewIncome"

    #3 kỹ thuật
    #các cột dương
    results = []
    positive_cols = [col1, col2]
    negative_cols = [col3]

    for col in positive_cols:
        original_skew = df[col].skew()

        # Log
        log_data = np.log1p(df[col])
        log_skew = log_data.skew()

        # Box-Cox
        boxcox_data, lam = boxcox(df[col])
        boxcox_skew = pd.Series(boxcox_data).skew()

        results.append([col, original_skew, log_skew, boxcox_skew, None, lam])

    #cột âm
    for col in negative_cols:
        before = df[col].skew()

        pt = PowerTransformer(method="yeo-johnson")
        power_data = pt.fit_transform(df[[col]]).flatten()

        skew_power = pd.Series(power_data).skew()
        lam = pt.lambdas_[0]

        results.append([col, before, None, None, skew_power, lam])

    #biểu đồ trước
    plt.figure(figsize=(15,8))

    # Price
    plt.subplot(2,3,1)
    sns.histplot(df["SalePrice"], kde=True)
    plt.title("SalePrice - Before")

    plt.subplot(2,3,2)
    sns.histplot(np.log1p(df["SalePrice"]), kde=True)
    plt.title("SalePrice - Log")

    plt.subplot(2,3,3)
    sns.histplot(boxcox(df["SalePrice"])[0], kde=True)
    plt.title("SalePrice - BoxCox")

    # LotArea
    plt.subplot(2,3,1)
    sns.histplot(df["LotArea"], kde=True)
    plt.title("LotArea - Before")

    plt.subplot(2,3,2)
    sns.histplot(np.log1p(df["LotArea"]), kde=True)
    plt.title("LotArea - Log")

    plt.subplot(2,3,3)
    sns.histplot(boxcox(df["LotArea"])[0], kde=True)
    plt.title("LotArea - BoxCox")

    # NegSkewIncome
    plt.subplot(2,3,4)
    sns.histplot(df["NegSkewIncome"], kde=True)
    plt.title("NegSkewIncome - Before")

    plt.subplot(2,3,5)
    sns.histplot(power_data.flatten(), kde=True)
    plt.title("NegSkewIncome - YeoJohnson")

    plt.tight_layout()
    plt.show()
# B2()

def B3():
    num_cols = df.select_dtypes(include=np.number).columns
    print(num_cols)
    skew_table = df[num_cols].skew().sort_values(ascending=False)
    print(skew_table)

    target = "SalePrice"

    features = ["LotArea", "Rooms"]

    X = df[features]
    y = df[target]

    # Chia dataset thành train/test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    
    #version A
    model_raw = LinearRegression()
    model_raw.fit(X_train, y_train)

    pred_raw = model_raw.predict(X_test)

    rmse_raw = np.sqrt(mean_squared_error(y_test, pred_raw))
    r2_raw = r2_score(y_test, pred_raw)

    #version B
    y_train_log = np.log1p(y_train)

    model_log = LinearRegression()
    model_log.fit(X_train, y_train_log)

    pred_log = model_log.predict(X_test)

    # đổi ngược về giá trị thật
    pred_log_inverse = np.expm1(pred_log)

    rmse_log = np.sqrt(mean_squared_error(y_test, pred_log_inverse))
    r2_log = r2_score(y_test, pred_log_inverse)

    #version C
    pt = PowerTransformer()

    X_train_power = pt.fit_transform(X_train)
    X_test_power = pt.transform(X_test)

    model_power = LinearRegression()
    model_power.fit(X_train_power, y_train)

    pred_power = model_power.predict(X_test_power)

    rmse_power = np.sqrt(mean_squared_error(y_test, pred_power))
    r2_power = r2_score(y_test, pred_power)

    #so sanhs kết quả
    result = pd.DataFrame({
    "Model": ["Raw Data", "Log Target", "Power Features"],
    "RMSE": [rmse_raw, rmse_log, rmse_power],
    "R2 Score": [r2_raw, r2_log, r2_power]
    })

    print(result)
# B3()

def B4():
    num_cols = df.select_dtypes(include=np.number).columns
    print(num_cols)
    skew_table = df[num_cols].skew().sort_values(ascending=False)
    print(skew_table)

    #chọn cột
    col1 = "SalePrice"
    col2 = "LotArea"

    #log
    df["price_log"] = np.log1p(df["SalePrice"])
    df["area_log"] = np.log1p(df["LotArea"])

    #version A
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    sns.histplot(df["SalePrice"], kde=True)
    plt.title("Raw Price")

    plt.subplot(1,2,2)
    sns.histplot(df["LotArea"], kde=True)
    plt.title("Raw Area")

    plt.tight_layout()
    plt.show()

    #version B
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    sns.histplot(df["price_log"], kde=True)
    plt.title("Log Price")

    plt.subplot(1,2,2)
    sns.histplot(df["area_log"], kde=True)
    plt.title("Log Area")

    plt.tight_layout()
    plt.show()

    #Tạo metric mới
    df["log_price_index"] = df["price_log"] / df["area_log"]

    print(df[["SalePrice", "LotArea", "log_price_index"]].head())
B4()