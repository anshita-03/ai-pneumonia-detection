from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tempfile
import datetime
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("AI-Based Pneumonia Detection System")
st.caption("Developed by Anshita")
st.warning(
    "This application is for educational purposes only and must not be used for real medical diagnosis."
)
st.markdown(
    "Detect signs of pneumonia from chest X-ray images using deep learning."
)
uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/model.h5')
    return model


with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']


def create_pdf(predicted_class, confidence):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    c = canvas.Canvas(temp_file.name, pagesize=letter)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, "Chest X-ray Analysis Report")

    c.setFont("Helvetica", 12)

    c.drawString(50, 700, f"Prediction: {predicted_class}")
    c.drawString(50, 675, f"Confidence: {confidence:.2f}%")

    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    c.drawString(50, 650, f"Generated on: {current_time}")

    c.drawString(50, 600, "Disclaimer:")
    c.drawString(
        50,
        580,
        "This tool is for educational purposes only."
    )

    c.save()

    return temp_file.name


def decode_img(image):

    img = tf.image.decode_jpeg(image, channels=3)

    img = tf.image.resize(img, [224, 224])

    img = img / 255.0

    return np.expand_dims(img, axis=0)

if uploaded_file is not None:
    content = uploaded_file.read()

    with st.spinner("Classifying image..."):
        prediction = model.predict(decode_img(content))

        label = np.argmax(prediction, axis=1)

        predicted_class = classes[label[0]]

        confidence = prediction.max() * 100

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

    pdf_path = create_pdf(predicted_class, confidence)

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Report",
            data=pdf_file,
            file_name="pneumonia_report.pdf",
            mime="application/pdf"
        )

    image = Image.open(BytesIO(content))

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_column_width=True
    )