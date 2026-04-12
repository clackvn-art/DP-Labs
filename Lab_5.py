import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def sm():
    df = pd.read_csv("Supermarket.csv")
    print(df.describe())
    print(df.isnull().sum())

    #datetime & index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    #missing value Forward Fill / Backward & Fill / Interpolation.
    # Forward Fill
    df = df.ffill()

    # Backward Fill
    df = df.bfill()

    # Interpolation
    df["revenue"] = df["revenue"].interpolate()

    #đặc trưng thời gian
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    #tổng doanh thu
    monthly_sales = df["revenue"].resample("ME").sum()
    weekly_sales = df["revenue"].resample("W").sum()

    monthly_sales.plot(title="Monthly Sales")
    plt.show()

    weekly_sales.plot(title="Weekly Sales")
    plt.show()

    # trend & seasonality
    df["rolling_30"] = df["revenue"].rolling(30).mean()

    df[["revenue","rolling_30"]].plot(figsize=(10,5))
    plt.show()
    print(df.isnull().sum())
# sm()

def web():
    wf = pd.read_csv("Web.csv")
    print(wf.describe())
    print(wf.isnull().sum())

    #tần suất dữ liệu
    wf["datetime"] = pd.to_datetime(wf["datetime"])
    wf = wf.set_index("datetime")
    wf = wf.asfreq("h")

    #missing value bằng interpolate
    wf["visits"] = wf["visits"].interpolate()
    
    #đặc trưng thời gian
    wf["hour"] = wf.index.hour
    wf["dayofweek"] = wf.index.dayofweek

    #biểu đồ
    hourly_avg = wf.groupby("hour")["visits"].mean()

    hourly_avg.plot(kind="bar", title="Visits by Hour")
    plt.show()

    #seasonality
    daily = wf["visits"].resample("D").mean()
    weekly = wf["visits"].resample("W").mean()

    daily.plot(title="Daily Seasonality")
    plt.show()

    weekly.plot(title="Weekly Seasonality")
    plt.show()
    print(wf.isnull().sum())
# web()
    
def stock():
    sf = pd.read_csv("Stock.csv")
    print(sf.describe())
    print(sf.isnull().sum())

    #datetime & index
    sf["date"] = pd.to_datetime(sf["date"])
    sf = sf.set_index("date")

    #missing value Forward Fill
    sf["close_price"] = sf['close_price'].ffill()

    #biểu đồ close price
    sf["close_price"].plot(title="Close Price")
    plt.show()

    #trend
    sf["ma7"] = sf["close_price"].rolling(7).mean()
    sf["ma30"] = sf["close_price"].rolling(30).mean()

    sf[["close_price","ma7","ma30"]].plot(figsize=(10,5))
    plt.show()

    #seasonality
    monthly_pattern = sf.groupby(sf.index.month)["close_price"].mean()

    monthly_pattern.plot(kind="bar", title="Monthly Pattern")
    plt.show()
# stock()

def product():
    pf = pd.read_csv("Production.csv")
    print(pf.describe())
    print(pf.isnull().sum())

    #datetime & index
    pf["week_start"] = pd.to_datetime(pf["week_start"])
    pf = pf.set_index("week_start")

    #missing value bằng Interpolation
    pf["production"] = pf["production"].interpolate()

    #đặc trưng thời gian
    pf["week"] = pf.index.isocalendar().week
    pf["quarter"] = pf.index.quarter
    pf["year"] = pf.index.year

    #trend
    pf["rolling_12"] = pf["production"].rolling(12).mean()

    pf[["production","rolling_12"]].plot(figsize=(10,5))
    plt.show()

    #seasonality
    quarter_avg = pf.groupby("quarter")["production"].mean()

    quarter_avg.plot(kind="bar", title="Quarterly Seasonality")
    plt.show()
    
    #decomposition
    result = seasonal_decompose(pf["production"], model="additive", period=12)
    result.plot()
    plt.show()
    print(pf.isnull().sum())
# product()