# VorobyovStas_241_AI_PR13 sales_forecast.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gplearn.genetic import SymbolicRegressor
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
if not hasattr(np, 'int'):
    np.int = int

# ======================
# 1. Відтворюваність
# ======================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ======================
# 2. Генерація синтетичних даних
# ======================
def generate_synthetic_sales(n_days=365):
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    trend = np.linspace(50, 100, n_days)  # тренд
    seasonality = 10 * np.sin(2 * np.pi * dates.dayofyear / 7)  # тижнева сезонність
    noise = np.random.normal(0, 5, n_days)  # випадковий шум
    promo = np.random.choice([0, 1], size=n_days, p=[0.8, 0.2])
    holiday = np.random.choice([0, 1], size=n_days, p=[0.9, 0.1])
    sales = trend + seasonality + noise + promo*5 + holiday*8
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'promo': promo,
        'holiday': holiday
    })
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    return df

df = generate_synthetic_sales()
df.set_index('date', inplace=True)

# ======================
# 3. Базові лаги та ролінги
# ======================
for lag in [1,7,28]:
    df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
for window in [7,28]:
    df[f'rolling_mean_{window}'] = df['sales'].shift(1).rolling(window).mean()
    df[f'rolling_std_{window}'] = df['sales'].shift(1).rolling(window).std()

df.dropna(inplace=True)

# ======================
# 4. Train/Validation/Test
# ======================
n = len(df)
train = df.iloc[:int(0.7*n)]
val = df.iloc[int(0.7*n):int(0.85*n)]
test = df.iloc[int(0.85*n):]

FEATURES = [c for c in df.columns if c != 'sales']
TARGET = 'sales'

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ======================
# 5. Базові лінії
# ======================
# Naive
naive_pred = test['sales_lag_1']
mae_naive = mean_absolute_error(y_test, naive_pred)
rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))
mape_naive = np.mean(np.abs((y_test - naive_pred)/y_test))*100
print("Baseline Naive:", round(mae_naive,2), round(rmse_naive,2), round(mape_naive,2), "%")

# Seasonal Naive (7 днів)
seasonal_pred = test['sales_lag_7']
mae_seasonal = mean_absolute_error(y_test, seasonal_pred)
rmse_seasonal = np.sqrt(mean_squared_error(y_test, seasonal_pred))
mape_seasonal = np.mean(np.abs((y_test - seasonal_pred)/y_test))*100
print("Baseline Seasonal:", round(mae_seasonal,2), round(rmse_seasonal,2), round(mape_seasonal,2), "%")

# ======================
# 6. ML: RandomForest
# ======================
rf = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape_rf = np.mean(np.abs((y_test - y_pred_rf)/y_test))*100
print("RandomForest:", round(mae_rf,2), round(rmse_rf,2), round(mape_rf,2), "%")

# ======================
# 7. GP: Symbolic Regressor
# ======================
gp = SymbolicRegressor(population_size=500,
                       generations=20,
                       stopping_criteria=0.01,
                       p_crossover=0.7,
                       p_subtree_mutation=0.2,
                       p_hoist_mutation=0.05,
                       p_point_mutation=0.05,
                       parsimony_coefficient=0.01,
                       random_state=RANDOM_SEED)

gp.fit(X_train.values, y_train.values)
y_pred_gp = gp.predict(X_test.values)

mae_gp = mean_absolute_error(y_test, y_pred_gp)
rmse_gp = np.sqrt(mean_squared_error(y_test, y_pred_gp))
mape_gp = np.mean(np.abs((y_test - y_pred_gp)/y_test))*100
print("GP:", round(mae_gp,2), round(rmse_gp,2), round(mape_gp,2), "%")

# ======================
# 8. Графік прогнозу
# ======================
plt.figure(figsize=(12,6))
plt.plot(test.index, y_test, label='Actual') #Фактичний
plt.plot(test.index, y_pred_rf, label='RF Prediction') #RF (ML) Прогноз
plt.plot(test.index, y_pred_gp, label='GP Prediction') #GP Прогноз
plt.plot(test.index, naive_pred, label='Naive', alpha=0.5)
plt.legend()
plt.title("Sales Forecasting") #Прогнозування продажів
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# ======================
# 9. Формула GP
# ======================
print("Best GP Formula:")
print(gp._program)
