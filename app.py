import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
import class_names  # File chứa dictionary classes

# Cấu hình trang
st.set_page_config(
    page_title="Nhận diện biển báo giao thông",
    page_icon="🚦",
    layout="wide"
)

# Tiêu đề
st.title("🚦 Ứng dụng Nhận diện Biển báo Giao thông")
st.markdown("---")

# Tải model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/traffic_model.h5')
        return model
    except:
        st.error("Không tìm thấy model! Vui lòng train model trước.")
        return None

model = load_model()

# Sidebar
st.sidebar.title("Tùy chọn")
option = st.sidebar.radio(
    "Chọn chế độ:",
    ["📤 Upload ảnh", "📷 Chụp ảnh", "ℹ️ Thông tin model"]
)

def preprocess_image(image):
    """Tiền xử lý ảnh đầu vào"""
    # Chuyển đổi sang numpy array
    image = np.array(image)
    # Resize về 32x32
    image = cv2.resize(image, (32, 32))
    # Normalize
    image = image / 255.0
    return image

def predict_image(image, model):
    """Dự đoán ảnh"""
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence, prediction[0]

if option == "📤 Upload ảnh":
    st.header("Upload ảnh biển báo")
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh biển báo giao thông", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Ảnh đã upload", use_column_width=True)
        
        with col2:
            if model is not None:
                with st.spinner("Đang nhận diện..."):
                    predicted_class, confidence, all_probs = predict_image(image, model)
                
                # Hiển thị kết quả
                st.success("✅ Nhận diện hoàn tất!")
                st.write(f"**Biển báo:** {class_names.classes[predicted_class]}")
                st.write(f"**Độ tin cậy:** {confidence*100:.2f}%")
                st.write(f"**Mã biển báo:** {predicted_class}")
                
                # Hiển thị top 5 dự đoán
                st.subheader("Top 5 dự đoán:")
                top5_indices = np.argsort(all_probs)[-5:][::-1]
                for i, idx in enumerate(top5_indices):
                    st.write(f"{i+1}. {class_names.classes[idx]} - {all_probs[idx]*100:.2f}%")

elif option == "📷 Chụp ảnh":
    st.header("Chụp ảnh biển báo")
    
    camera_image = st.camera_input("Chụp ảnh biển báo giao thông")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Ảnh đã chụp", use_column_width=True)
        
        with col2:
            if model is not None:
                with st.spinner("Đang nhận diện..."):
                    predicted_class, confidence, all_probs = predict_image(image, model)
                
                st.success("✅ Nhận diện hoàn tất!")
                st.write(f"**Biển báo:** {class_names.classes[predicted_class]}")
                st.write(f"**Độ tin cậy:** {confidence*100:.2f}%")

else:
    st.header("Thông tin Model")
    
    st.subheader("📊 Kiến trúc Model")
    st.code("""
    Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    """)
    
    st.subheader("📈 Hiệu suất")
    st.metric("Độ chính xác trên tập test", "99.55%")
    
    st.subheader("📋 Các lớp biển báo")
    # Hiển thị một số lớp biển báo
    for i in range(0, 43, 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < 43:
                with col:
                    st.text(f"{i+j}: {class_names.classes[i+j]}")

# Footer
st.markdown("---")
st.markdown("Ứng dụng Nhận diện Biển báo Giao thông sử dụng CNN")