import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Caminho do arquivo
arquivo_csv = r'C:\Users\grizzi\05-PycharmProjects\Climaforecast\01-2022_04-2023.csv'

# Ler o arquivo CSV
df = pd.read_csv(arquivo_csv)

    ##Pré-processamento dos dados:

# Converte a coluna "datetime" para o tipo datetime do Pandas
df['datetime'] = pd.to_datetime(df['datetime'])

# Extrai o mês e ano da coluna "datetime" e cria colunas separadas
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Agrupa os dados por mês e ano, calculando a média da precipitação
df_monthly = df.groupby(['year', 'month']).agg({'precipcover': 'mean'}).reset_index()

    ## Divisão dos dados em conjuntos de treinamento e teste

X = df_monthly[['year', 'month']]
y = df_monthly['precipcover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Treinando o modelo de regressão, como o RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

    ## Avaliando a performance do modelo usando o conjunto de teste:

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

    ## Previsão de chuvas para cada mês do ano desejado:

ano_desejado = 2024
meses = list(range(1, 13))

previsoes = model.predict([[ano_desejado, mes] for mes in meses])

for mes, prev in zip(meses, previsoes):
    print(f"Previsão de chuva para {ano_desejado}-{mes:02d}: {prev:.2f}")

    ## Plotar os dados em um gráfico

def plotar_previsoes(ano_desejado, meses, previsoes):
    plt.figure(figsize=(10, 6))
    plt.plot(meses, previsoes, marker='o', linestyle='-', label=f"Previsão {ano_desejado}")
    plt.xlabel('Mês')
    plt.ylabel('Precipitação Média')
    plt.title(f'Previsão de Chuva para {ano_desejado}')
    plt.xticks(meses, labels=[f"{m:02d}" for m in meses])
    plt.legend()
    plt.grid(True)
    plt.show()

plotar_previsoes(ano_desejado, meses, previsoes)




