# Import thư viện cần thiết
import streamlit as st
import numpy as np
import joblib

# Tải mô hình đã huấn luyện
model = joblib.load('linear_regression_model.pkl')

# Tạo giao diện
st.title('Dự đoán giá trị')

# Tạo các trường nhập liệu cho các đặc trưng
feature1 = st.number_input('Feature 1', min_value=0.0, max_value=1.0, value=0.5)
feature2 = st.number_input('Feature 2', min_value=0.0, max_value=1.0, value=0.5)
feature3 = st.number_input('Feature 3', min_value=0.0, max_value=1.0, value=0.5)

# Tạo nút dự đoán
if st.button('Dự đoán'):
    features = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(features)
    st.write(f'Giá trị dự đoán: {prediction[0]}')
