import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
import pandas as pd
from datetime import datetime

if 'page' not in st.session_state:
    st.session_state['page'] = 'prediction'

model_1 = tf.keras.models.load_model('bat.h5')
conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='tb_test_results_db'
        )
cursor = conn.cursor()


def predict_tb(image):

    image = np.array(image)


    if len(image.shape) == 2:

        gray_image = image
    elif len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unexpected image format")

    resized_image = cv2.resize(gray_image, (224, 224))
    resized_image = resized_image / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)
    prediction = model_1.predict(resized_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_labels = {0: 'No TB', 2: 'TB'}
    result = class_labels.get(predicted_class, 'Unknown')

    return result,prediction


def store_result_in_db(name, dob, result,now):
    try:

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tb_test_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                dob DATE,
                result VARCHAR(255),
                now VARCHAR(255)
            )
        ''')

        cursor.execute('''
            INSERT INTO tb_test_results (name, dob, result, now)
            VALUES (%s, %s, %s, %s)
        ''', (name, dob, result, now))

        conn.commit()

    except mysql.connector.Error as err:
        st.write(f"Error: {err}")
    finally:
        # cursor.close()
        # conn.close()

def get_previous_tests(name, dob):
    try:
        # conn = mysql.connector.connect(
        #     host='localhost',
        #     user='root',
        #     password='636396',
        #     database='tb_test_results_db'
        # )
        # cursor = conn.cursor()

        query = "SELECT * FROM tb_test_results WHERE name = %s AND dob = %s"
        cursor.execute(query, (name, dob))
        rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=['ID', 'Name', 'Date of Birth', 'Result', 'Date and Time'])
        return df

    except mysql.connector.Error as err:
        st.write(f"Error: {err}")
        return None
    finally:
        # cursor.close()
        # conn.close()

def get_all_results_from_db():
    try:

        # conn = mysql.connector.connect(
        #     host='localhost',
        #     user='root',
        #     password='636396',
        #     database='tb_test_results_db'
        # )
        # cursor = conn.cursor()
        cursor.execute('SELECT * FROM tb_test_results')
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=['ID', 'Name', 'Date of Birth', 'Result', 'Date and Time'])

        return df

    except mysql.connector.Error as err:
        st.write(f"Error: {err}")
        return None
    finally:
        # cursor.close()
        # conn.close()

if st.session_state.page == 'prediction':
    st.title('Tuberculosis Test Result')

    name = st.text_input("Patient Name")
    dob = st.date_input("Date of Birth")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Get Tuberculosis Test Result'):
            result, prediction = predict_tb(image)
            now = datetime.now()
            st.write(f"Result: {result}")
            st.write(f"Prediction probabilities: {prediction}")
            store_result_in_db(name, dob, result, now)
            st.write("Data stored in the database successfully!")

    if st.sidebar.button('Show Database'):
        st.session_state.page = 'database'
        st.rerun()
    if st.button('Previous Test'):
        previous_tests_df = get_previous_tests(name, dob)
        if previous_tests_df is not None and not previous_tests_df.empty:
            st.write("Previous Tests:")
            st.dataframe(previous_tests_df)
        else:
            st.write("No previous tests found.")

elif st.session_state.page == 'database':
    st.title('Database Records')

    df = get_all_results_from_db()
    if df is not None:
        st.write("Complete Database:")
        st.dataframe(df)

    if st.sidebar.button('Back to Prediction'):
        st.session_state.page = 'prediction'
        st.rerun()
