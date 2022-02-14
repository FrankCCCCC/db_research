# %%
import utility
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def weighted_mean(x):
    arr = np.ones((1, x.shape[1]))
    arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
    return arr

def make_window(x_feats, window=10):
    def count_(x):
        # print(x)
        display(x_feats.loc[x.index])
        now = x[-1, 0]
        return sum([x[:, 0] - now < 1000])

    window_col = x_feats['Start Time'].rolling(window, min_periods=0).apply(count_, raw=False)
    # window_col = x_feats.rolling(2, method="table", min_periods=0).apply(weighted_mean, raw=True, engine="numba")
    print(window_col.shape)
    x_feats['window'] = window_col
    return x_feats

def split_train_test(X, y, sel_col_names, test_size=0.33, is_all_feats=True):
    if sel_col_names is not None:
        y_input = y[sel_col_names]
        if not is_all_feats:
            X_input = X[sel_col_names]
        else:
            X_input = X
    else:
        y_input = y

    X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=test_size)
    return X_train, X_test, y_train, y_test

def training(X_train, y_train):
    random_state = 42
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state)
    rfr.fit(X_train, y_train)

    return rfr

def evaluate(X_test, y_test, model):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MSE: {mse}, MAPE: {mape}")
    return y_pred, mse, mape

# %%
if __name__ == '__main__':
    df_ou7_feats, df_ou7_lats = utility.get_df_join(ou_id=7)

    print("OU - 7 Features")
    display(df_ou7_feats)

    print("OU - 7 Latencies")
    display(df_ou7_lats)
    df_ou7_feats = make_window(df_ou7_feats, window=10)

    ou_names = ['OU7 - Write to Local', 
                'OU7 - Flush - Pass To The Next Tx', 
                'OU7 - Flush - Get Record', 
                'OU7 - Flush - Insert To Cache', 
                'OU7 - Flush - Write Back', 
                'OU7 - Core Update', 
                'OU7 - sLockIndex', 
                'OU7 - Core Insert', 
                'OU7 - xLockIndex', 
                'OU7 - Flush - Take From Tx']
    for ou_name in ou_names:
        X_train, X_test, y_train, y_test = split_train_test(df_ou7_feats, df_ou7_lats, sel_col_names=ou_name, is_all_feats=True)
        model = training(X_train, y_train)
        print(ou_name)
        evaluate(X_test, y_test, model)

# %%
