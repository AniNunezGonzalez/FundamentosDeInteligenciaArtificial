import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos desde el CSV
data = pd.read_csv('BaseMatrizDeRiesgosNumerica.csv')

# Verificar los primeros registros del dataframe
print(data.head())

# "IncumplimientoHistorico" es la variable objetivo
features = ['Edad', 'IngresoAnual', 'Producto', 'IntencionPrestamo', 'MontoPrestamo', 'TasaInteres', 'PorcentajeIngresos', 'HistorialCrediticio']
X = data[features]
y = data['IncumplimientoHistorico']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Realizar predicciones
predictions = model.predict(X_test)

# Umbral para decidir si es riesgoso 
threshold = 0.5
risk_labels = (predictions > threshold).astype(int)


# Revertir la normalización (si es necesario, y si guardaste el scaler original)
X_test_original = scaler.inverse_transform(X_test)

# Crear un DataFrame con las características originales y las predicciones
result = pd.DataFrame(X_test_original, columns=features)
result['Riesgo'] = risk_labels.flatten()

# Mostrar los resultados
pd.set_option('display.max_rows', None)
print(result)



