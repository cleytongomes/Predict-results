# importando as bibliotecas
import tensorflow as tf
import numpy as np

# Craindo as listas de treinamento
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float) # Entrada
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float) # Saída

# Imprimindo os valores nas escalas
for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# Criando um layer de entrada
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# O modelo recebe uma lista de layers como argumento
model = tf.keras.Sequential([l0])

# Compilando o modelo
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
print("Finished training the model")

# Predizendo um valor de entrada
entrada = float(input())
print(model.predict([entrada]))