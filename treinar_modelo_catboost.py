import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregar o conjunto de dados fictício
df = pd.read_csv('dados_ficticios_churn.csv')

# Separar features (X) e target (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identificar as variáveis categóricas
cat_features = ['genero', 'localizacao', 'frequencia_uso', 'historico_pagamento']

# Criar Pools de dados para o CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Inicializar e treinar o modelo CatBoost
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
model.fit(train_pool)

# Fazer previsões
y_pred = model.predict(test_pool)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Analisar a importância das features
importances = model.get_feature_importance()
feature_names = X_train.columns

# Criar um DataFrame para exibir a importância das features
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Ordenar o DataFrame pela importância
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)
