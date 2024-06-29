import streamlit as st 
import requests
from streamlit_lottie import st_lottie
from PIL import Image

st.set_page_config(
    page_title="Monelytics",
    page_icon="💸",
)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_encoded = get_image_base64('images/Monelytics.png')  # Path to the uploaded image

# st.sidebar.markdown("""
#     <div style="margin-top: 10px; margin-bottom: 50px; display: flex; justify-content: center;">
#         <img src="data:image/png;base64,{}" style="width: 50%;">
#     </div>
#     """.format(image_encoded), unsafe_allow_html=True)

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://lottie.host/308bdfd2-e2bd-48a8-be92-21cf7af8b9bb/Bq1XF0xkWw.json")
img_verren = Image.open("images/verren.png")
img_calvin = Image.open("images/calvin.jpeg")
img_marvella = Image.open("images/marvella.jpeg")
img_dean = Image.open("images/dean.png")

# ---- HEADER SECTION ----
with st.container():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{}" style="width: 300px; margin-right: 20px;">
        <div>
            <h1 style="margin-bottom: 10px;">💹 MONELYTICS</h1>
            <h3 style="margin-bottom: 10px;">Stock Prediction for the Four Largest Banks in Indonesia</h3>
            <div style="background-color: #32CD32; padding: 10px; border-radius: 5px; color: white;">
                Monelytics is an innovative solution for stock predictions, especially for the four largest banks in Indonesia. Leveraging advanced data analysis technology, Monelytics provides deep insights into market trends and stock performance predictions, helping investors make better investment decisions.
            </div>
        </div>
    </div>
    """.format(image_encoded), unsafe_allow_html=True)

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("📊 Why Choose Monelytics?")
        st.write("##")
        st.write(
            """
            - Uses various machine learning and deep learning algorithms to analyze historical patterns and current trends, allowing for accurate predictions of future stock movements.
            - Easy-to-use interface offering clear and comprehensive data visualizations, including stock movement charts, trading volumes, and other key indicators.
            - Relied upon for its high prediction accuracy, helping you avoid unprofitable investment decisions.

            Discover your next investment opportunity with Monelytics – your trusted companion for navigating the Indonesian stock market.
            """
        )
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("👩🏻‍💻Our Developers")
    st.write("###")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_dean, width=170)
    with text_column:
        st.subheader("Dean Hans Felandio Saputra")
        st.write(
            """
            - Logistic Regressor and Random Forest Model Developer
            - Data Engineering 1
            """
        )
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_calvin, width=170)
    with text_column:
        st.subheader("Calvin Alexander")
        st.write(
            """
            - LSTM and CNN Model Developer
            - Stock Consultant
            """
        )
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_marvella, width=170)
    with text_column:
        st.subheader("Marvella Shera Devi")
        st.write(
            """
            - Support Vector Regressor and ARIMA Model Developer
            - Document Administrator
            """
        )
with st.container():
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_verren, width=170)
    with text_column:
        st.subheader("Verren Angelina Saputra")
        st.write(
            """
            - ANN, Decision Tree Regressor, and ANN Model Developer
            - Data Engineering 2 and Full Stack Developer
            """
        )

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("📩 Get In Touch With Monelytics!")
    st.write("##")

    # Documentation: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
