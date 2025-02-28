# import streamlit as st
# import numpy as np
# import joblib
#
# from train import accuracy
#
# model = joblib.load('model/model.pkl')
# st.title = ('Breast Canser Diagnosis')
# input_data = st.text_input('input data')
#
# if st.button('Predict'):
#     # inputdf = np.array(float(input_data))
#     # prediction = model.predict(inputdf.reshape(1, -1))
#
#     # Convert the input string to a NumPy array of floats
#     input_list = [float(i) for i in input_data.split(',')]
#
#     # Reshape to match the model's expected input shape
#     inputdf = np.array(input_list).reshape(1, -1)
#
#     # Making prediction
#     prediction = model.predict(inputdf)
#     if prediction =='M':
#         result = "Malignant"
#     if prediction == "B":
#         result = "Benign"
#     else :
#         result = "Error: Invalid input"
#     st.success(f"The predicted result is {result},with the accuracy of {accuracy}")

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model/model.pkl')

# Input form for the features (comma-separated)
st.title('Breast Cancer Diagnosis')
input_data = st.text_input('Enter features as comma-separated values (e.g., 17.99,10.38,122.8,1001,...)')

if st.button('Predict'):
    try:
        # Convert the input string into a list of floats
        input_list = [float(i) for i in input_data.split(',')]

        # Ensure the list has 30 features
        if len(input_list) != 30:
            st.error("Invalid input: Please enter exactly 30 features.")
        else:
            # Convert the list into a numpy array and reshape it to match the model's input
            inputdf = np.array(input_list).reshape(1, -1)

            # Make the prediction
            prediction = model.predict(inputdf)

            # Map the prediction to 'Malignant' or 'Benign'
            if prediction == 'M':
                result = "Malignant"
            elif prediction == 'B':
                result = "Benign"
            else:
                result = "Error: Invalid prediction result"

            # Display the result
            accuracy = 0.9298245614035088  # Or get this dynamically if available
            st.success(f"The predicted result is {result}, with the accuracy of {accuracy * 100:.2f}%")
    except ValueError:
        st.error("Invalid input: Please make sure the input consists of numerical values separated by commas.")
