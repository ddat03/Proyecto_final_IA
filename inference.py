import joblib

# Datos fijos para prueba
features = [[53, 1, 1, 0, 22.500]]
# Cargar el modelo
modelo = joblib.load("dtc_model.pkl")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])
