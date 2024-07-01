import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Función para cargar datos y entrenar el modelo
def train_model_and_predict_risk():
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

    # Guardar el scaler para uso posterior
    joblib.dump(scaler, 'scaler.pkl')

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

    # Guardar el modelo entrenado
    model.save('modelo_riesgo.h5')

    return model, scaler

# Función para predecir el riesgo de un nuevo cliente
def predecir_riesgo(model, scaler, nuevo_cliente):
    # Convertir el nuevo cliente a un array y escalarlo
    nuevo_cliente_array = np.array(nuevo_cliente).reshape(1, -1)
    nuevo_cliente_scaled = scaler.transform(nuevo_cliente_array)

    # Realizar la predicción
    prediccion = model.predict(nuevo_cliente_scaled)

    # Aplicar el umbral
    threshold = 0.5
    riesgo = (prediccion > threshold).astype(int)

    return riesgo[0][0]

# Función principal para cargar el modelo y realizar predicciones
def main():
    # Cargar el modelo y el scaler
    try:
        model = load_model('modelo_riesgo.h5')
        scaler = joblib.load('scaler.pkl')
    except (OSError, IOError) as e:
        print("No se encontró el modelo entrenado. Entrenando un nuevo modelo...")
        model, scaler = train_model_and_predict_risk()

    # Solicitar datos del nuevo cliente
    print("\nIngrese los datos del nuevo cliente:")
    edad = float(input("Edad: "))
    ingreso_anual = float(input("Ingreso Anual: "))
    producto = float(input("Producto: "))
    intencion_prestamo = float(input("Intención de Préstamo: "))
    monto_prestamo = float(input("Monto del Préstamo: "))
    tasa_interes = float(input("Tasa de Interés: "))
    porcentaje_ingresos = float(input("Porcentaje de Ingresos: "))
    historial_crediticio = float(input("Historial Crediticio: "))

    # Crear una lista con los datos del nuevo cliente
    nuevo_cliente = [edad, ingreso_anual, producto, intencion_prestamo, monto_prestamo, tasa_interes, porcentaje_ingresos, historial_crediticio]

    # Predecir el riesgo
    riesgo = predecir_riesgo(model, scaler, nuevo_cliente)

    # Mostrar el resultado
    if riesgo == 1:
        print("El cliente tiene un alto riesgo de incumplimiento.")
    else:
        print("El cliente tiene un bajo riesgo de incumplimiento.")

if __name__ == "__main__":
    main()
