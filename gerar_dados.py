import pandas as pd
import numpy as np

# Definindo o número de amostras
num_samples = 1000

# Criando um DataFrame fictício
np.random.seed(42)
data = {
    'idade': np.random.randint(18, 70, size=num_samples),
    'genero': np.random.choice(['Masculino', 'Feminino'], size=num_samples),
    'localizacao': np.random.choice(['Cidade A', 'Cidade B', 'Cidade C'], size=num_samples),
    'frequencia_uso': np.random.choice(['Baixa', 'Média', 'Alta'], size=num_samples),
    'duracao_contrato': np.random.randint(1, 24, size=num_samples),
    'historico_pagamento': np.random.choice(['Bom', 'Regular', 'Ruim'], size=num_samples),
    'churn': np.random.choice([0, 1], size=num_samples)  # 0: Não Churn, 1: Churn
}

df = pd.DataFrame(data)

# Salvando o DataFrame em um arquivo CSV
df.to_csv('dados_ficticios_churn.csv', index=False)

# Exibindo as primeiras linhas do DataFrame
print(df.head())
