# Utilizando SVR para fazer as previsões
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Ler os dados do arquivo CSV
df = pd.read_csv("C:\\Users\\grizzi\\05-PycharmProjects\\Climaforecast\\AI.csv")

# Converter a coluna 'Date' em um valor numérico
df['Date'] = pd.to_datetime(df['Date'])
min_date = df['Date'].min()
df['Date_numeric'] = (df['Date'] - min_date) / np.timedelta64(1, 'D')

# Selecionar colunas 'Date_numeric' e 'Close'
df = df[['Date_numeric', 'Close']]

# Remover ou preencher valores ausentes
df = df.dropna()

# Preparar os dados para treinamento e teste
X = df[['Date_numeric']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo SVR
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X_train, y_train)

# Avaliar o desempenho do modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Prever os preços das ações para as datas específicas
target_dates = ['2023-04-21', '2023-04-24', '2023-04-25', '2023-04-26', '2023-04-27', '2023-04-28']
target_dates_numeric = [(pd.to_datetime(date) - min_date) / np.timedelta64(1, 'D') for date in target_dates]
target_dates_numeric = np.array(target_dates_numeric).reshape(-1, 1)

predictions = model.predict(target_dates_numeric)

# Exibir as previsões
for date, pred in zip(target_dates, predictions):
    print(f"{date}: {pred}")

