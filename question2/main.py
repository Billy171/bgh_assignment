import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from helpers import TradingStrategy


if __name__ == "__main__":
    df = pd.read_csv("crypto_train_research_scientist.csv")

    """preliminary data analysis"""
    plt.hist(df['Target'], bins=1000)
    plt.title('Crypto single day returns')
    plt.xlim(-1, 1)
    plt.show()

    """dealing with missing values"""
    df.fillna(df.median(), inplace=True)

    """order by date, split the data"""
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values('Date')

    n = 2000
    past_data = df.iloc[:n]
    current_data = df.iloc[n:-n]
    future_data = df.iloc[-n:]

    dat = current_data.drop('Date', axis=1)
    x = dat.drop('Target', axis=1)
    y = dat['Target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    """applying a simple random forest and ridge as baselines"""
    rf = RandomForestRegressor(max_depth=10, random_state=0, max_features='sqrt')
    ridge = Ridge(alpha=1.0)

    rf.fit(x_train, y_train)
    ridge.fit(x_train, y_train)

    mse_train_rf = np.mean((rf.predict(x_train)-y_train)**2)
    mse_test_rf = np.mean((rf.predict(x_test)-y_test)**2)
    mse_train_ridge = np.mean((ridge.predict(x_train)-y_train)**2)
    mse_test_ridge = np.mean((ridge.predict(x_test)-y_test)**2)
    print(mse_train_rf, mse_test_rf)
    print(mse_train_ridge, mse_test_ridge)

    """plt.hist(ridge.predict(x_train), bins=100)
    plt.show()
    plt.hist(ridge.predict(x_test), bins=100)
    plt.show()"""

    """optimisation"""
    """
    params_rf = {'n_estimators': [300, 500, 750],
                 'max_depth': [20, 30],
                 'max_features': ['sqrt']}
    params_ridge = {'alpha': np.arange(0.1, 1.1, 0.1)}

    rf = RandomForestRegressor()
    ridge = Ridge()

    rf_gs = GridSearchCV(rf, params_rf, scoring='neg_mean_squared_error')
    rf_gs.fit(x_train, y_train)
    print(rf_gs.best_score_)
    print(rf_gs.best_params_)

    optim_model_rf = RandomForestRegressor(n_estimators=rf_gs.best_params_['n_estimators'],
                                           max_depth=rf_gs.best_params_['max_depth'],
                                           max_features=rf_gs.best_params_['max_features'])
    optim_model_rf.fit(x_train, y_train)
    mse_train_rf = np.mean((optim_model_rf.predict(x_train) - y_train) ** 2)
    mse_test_rf = np.mean((optim_model_rf.predict(x_test) - y_test) ** 2)

    print('mse_train_rf = ', mse_train_rf)
    print('mse_test_rf = ', mse_test_rf)

    ridge_gs = GridSearchCV(ridge, params_ridge, scoring='neg_mean_squared_error')
    ridge_gs.fit(x_train, y_train)
    print(ridge_gs.best_score_)
    print(ridge_gs.best_params_)

    optim_model_ridge = Ridge(alpha = ridge_gs.best_params_['alpha'])
    optim_model_ridge.fit(x_train, y_train)
    mse_train_ridge = np.mean((optim_model_ridge.predict(x_train) - y_train) ** 2)
    mse_test_ridge = np.mean((optim_model_ridge.predict(x_test) - y_test) ** 2)

    print('mse_train_ridge = ', mse_train_ridge)
    print('mse_test_ridge = ', mse_test_ridge)
    """

    best_params = {'max_depth': 30, 'max_features': 'sqrt', 'n_estimators': 750}

    """optim_model_rf = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                           max_depth=best_params['max_depth'],
                                           max_features=best_params['max_features'])
    optim_model_rf.fit(x_train, y_train)
    pickle.dump(optim_model_rf, open('model.pkl', 'wb'))"""

    optim_model_rf = pickle.load(open('model.pkl', 'rb'))
    mse_train_rf = np.mean((optim_model_rf.predict(x_train) - y_train) ** 2)
    mse_test_rf = np.mean((optim_model_rf.predict(x_test) - y_test) ** 2)

    """backtesting"""
    start_sum = 100

    strat = TradingStrategy(optim_model_rf, 0.2, 'uniform', start_sum)
    returns_past = strat.run_trade_sim(past_data, 'past')
    returns_future = strat.run_trade_sim(future_data, 'future')

    """Sharpe Ratio"""
    r_past = np.array(returns_past[1:])/np.array(returns_past[:-1])
    r_past = np.delete(r_past, np.where(r_past == 1))
    s_past = r_past.mean() / r_past.std()
    print(r_past.mean(), r_past.std())

    r_future = np.array(returns_future[1:])/np.array(returns_future[:-1])
    r_future = np.delete(r_future, np.where(r_future == 1))
    s_future = r_future.mean() / r_future.std()

    print(s_past, s_future)


