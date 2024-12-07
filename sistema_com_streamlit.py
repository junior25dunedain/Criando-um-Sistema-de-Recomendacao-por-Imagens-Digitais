import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import Normalizer
import cv2

# Carregar o conjunto de dados CIFAR-10 e o modelo treinado
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Carregar o modelo treinado (substitua pelo seu modelo real se necessário)
model = tf.keras.models.load_model('meu_modelo.h5')  # Certifique-se de ter salvo o modelo anteriormente
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
features = feature_extractor.predict(x_test)

# Função para calcular a distância euclidiana
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# Função para encontrar as imagens mais similares
def find_similar_images(query_feature, features, top_n=5):
    distances = [euclidean_distance(query_feature, feature) for feature in features]
    similar_image_indices = np.argsort(distances)[:top_n]
    return similar_image_indices

# Configurar a página do Streamlit
st.title("Sistema de Recomendação de Imagens")
st.write("Faça upload de uma imagem para encontrar produtos similares por aparência.")

# Carregar a imagem de consulta
uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    # Processar a imagem carregada
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extrair características da imagem de consulta
    query_feature = feature_extractor.predict(img_array)

    # Encontrar as imagens mais similares
    similar_images_indices = find_similar_images(query_feature, features)

    st.write("Imagens Similares:")
    for idx in similar_images_indices:
        similar_image = x_test[idx]
        st.image(similar_image, caption=f"Imagem Similar {idx}", use_column_width=True)
