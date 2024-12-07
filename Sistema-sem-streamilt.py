import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Carregar o conjunto de dados CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar os valores dos pixels
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Codificação one-hot dos rótulos
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Conjunto de dados carregado e normalizado!")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Construir o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Modelo construído e compilado!")

# Treinar o modelo 
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test)) 
print("Modelo treinado!")

# Criar um modelo que termine na camada de Flatten
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)

# Extrair características das imagens de teste
features = feature_extractor.predict(x_test)

print("Características extraídas!")

import numpy as np

# Função para calcular a distância euclidiana
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# Encontrar as imagens mais similares
def find_similar_images(query_image_index, features, top_n=5):
    query_feature = features[query_image_index]
    distances = [euclidean_distance(query_feature, feature) for feature in features]
    similar_image_indices = np.argsort(distances)[:top_n]
    return similar_image_indices

# Exemplo de uso
similar_images = find_similar_images(0, features)
print("Índices das imagens mais similares:", similar_images)


from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Receber a imagem de consulta
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Extrair características
    query_feature = feature_extractor.predict(img)

    # Encontrar as imagens mais similares
    distances = [euclidean_distance(query_feature, feature) for feature in features]
    similar_image_indices = np.argsort(distances)[:5]

    # Retornar os resultados
    return jsonify(similar_images=similar_image_indices.tolist())

if __name__ == '__main__':
    app.run(debug=True)
