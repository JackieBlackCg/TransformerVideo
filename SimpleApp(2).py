import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import tensorflow as tf
import torch  # Import PyTorch
import keras
from keras.utils import register_keras_serializable
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
from keras.applications.densenet import DenseNet121


import streamlit as st
import os
import base64
# Ruta al archivo de la imagen
path_to_image = '/content/emo.webp'

# Tamaño deseado para el logo
logo_size = 50  # Puedes ajustar este tamaño según sea necesario

# Función para convertir la imagen a base64
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config (
    page_title = 'EmoTrans',
    page_icon = '🤖',
)        

# Personalizar la disposición usando markdown con HTML/CSS
st.markdown("""
    <style>
    .container {
        display: flex;
        align-items: center;
    }
    .logo {
        width: %spx;
        margin-right: 10px;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    </style>
    """ % logo_size, unsafe_allow_html=True)

# Usando raw HTML para mostrar el logo y el título con la disposición deseada
st.markdown(f"""
    <div class="container">
        <img class="logo" src="data:image/png;base64,{img_to_base64(path_to_image)}">
        <p class="title">EmoTrans</p>
    </div>
    """, unsafe_allow_html=True)


# CSS para mejorar el estilo de la app
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cargar el CSS
local_css("style.css")

# Colocar el logo en la barra lateral
st.sidebar.image(path_to_image, width=logo_size)
# Definir el menú de hamburguesa
menu_options = ["Home", "Cargar Video"]
menu_selection = st.sidebar.selectbox("Menu", menu_options)

# Página de inicio (Home)
if menu_selection == "Home":
    st.write("¡Bienvenido a la página de inicio!")



@register_keras_serializable(package='Custom', name='TransformerEncoder')
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation=keras.activations.gelu),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()



    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


@register_keras_serializable(package='Custom', name='PositionalEmbedding')
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        inputs = tf.cast(inputs, self.compute_dtype)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions


# Assuming feature_extractor, label_processor, and load_video are defined elsewhere

MAX_SEQ_LENGTH = 20  # Example value
NUM_FEATURES = 1024  # Example value
IMG_SIZE = 128       # Example value

# Load PyTorch model
model_path = 'trained_model.pth'  # Adjust the path as necessary
model = torch.load(model_path, map_location=torch.device('cpu'))  # Assuming you're using CPU

train_df = pd.read_csv("/content/train.csv")
# Label preprocessing with StringLookup.
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
)

center_crop_layer = layers.CenterCrop(IMG_SIZE, IMG_SIZE)

def build_feature_extractor():
    feature_extractor = DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy()  # Convierte el tensor a un array de numpy
    cropped = tf.squeeze(cropped).numpy()  # Elimina dimensiones de tamaño uno y convierte a numpy
    return cropped

def load_video(path, max_frames=0, offload_to_cpu=False):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frame = crop_center(frame)
            if offload_to_cpu and keras.backend.backend() == "torch":
                frame = frame.to("cpu")
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    if offload_to_cpu and keras.backend.backend() == "torch":
        return np.array([frame.to("cpu").numpy() for frame in frames])
    return np.array(frames)

def prepare_single_video(frames):
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate(frames, padding)

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0

    return frame_features


import os
import streamlit as st
import matplotlib.pyplot as plt

def predict_action(path):
    progress_bar = st.progress(0)
    class_vocab = label_processor.get_vocabulary()
    frames = load_video(os.path.join("test", path), offload_to_cpu=True)
    progress_bar.progress(25)  # Actualiza al 25% después de cargar el video

    frame_features = prepare_single_video(frames)
    progress_bar.progress(50)  # Actualiza al 50% después de procesar los frames

    probabilities = model.predict(frame_features)[0]
    progress_bar.progress(75)  # Actualiza al 75% después de hacer la predicción

    fig, ax = plt.subplots()
    bars = ax.bar(class_vocab, probabilities)
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval*100:.2f}%', va='bottom')

    st.pyplot(fig)
    progress_bar.progress(100)  # Completa la barra de progreso

    return frames


def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return "animation.gif"
from moviepy.editor import VideoFileClip
from tempfile import NamedTemporaryFile

st.title('Video Action Recognition')

uploaded_file = st.file_uploader("Choose a video...", type=["avi", "mp4"])

if uploaded_file is not None:
    # Crear un archivo temporal para guardar el video AVI
    with NamedTemporaryFile(delete=False, suffix='.avi') as tmp:
        tmp.write(uploaded_file.read())
        original_video_path = tmp.name

    # Determinar el tipo del archivo y procesar según sea necesario
    if uploaded_file.type == "video/x-msvideo":  # Tipo MIME para AVI
        # Convertir AVI a MP4 para visualización
        clip = VideoFileClip(original_video_path)
        mp4_path = original_video_path.replace('.avi', '.mp4')
        clip.write_videofile(mp4_path, codec='libx264')
        display_video_path = mp4_path
    else:
        # Si es MP4, usarlo directamente para la visualización
        display_video_path = original_video_path

    # Mostrar el video
    st.video(display_video_path)

    # Usar el archivo AVI para la predicción
    test_frames = predict_action(original_video_path)
    gif_path = to_gif(test_frames[:MAX_SEQ_LENGTH])
    st.image(gif_path)

st.sidebar.header('Parameters')
MAX_SEQ_LENGTH = st.sidebar.number_input('Max sequence length', value=20, step=1)
NUM_FEATURES = st.sidebar.number_input('Number of features', value=1024, step=1)
IMG_SIZE = st.sidebar.number_input('Image size', value=128, step=1)
