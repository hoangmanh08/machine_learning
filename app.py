import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Tiêu đề ứng dụng
st.title("Machine Learning Model Trainer")

# Tải lên file CSV
uploaded_file = st.file_uploader("D:\Hệ hỗ trợ quyết định\model\media prediction and its cost.csv", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Chọn đặc trưng và nhãn
    features = st.multiselect("Select Features", options=df.columns.tolist(), default=df.columns[:-1].tolist())
    label = st.selectbox("Select Label", options=df.columns.tolist(), index=len(df.columns) - 1)

    if features and label:
        X = df[features]
        y = df[label]

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Dự đoán và tính toán độ chính xác
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Hiển thị kết quả
        st.write(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")

        # Lưu mô hình nếu cần
        if st.button("Save Model"):
            import pickle
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)
            st.write("Model saved as 'trained_model.pkl'")
