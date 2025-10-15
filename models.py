# model.py (sửa lại phần load data)
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_synthetic_data():
    """Tạo dữ liệu giả để test nếu không có dataset thật"""
    print("🔄 Đang tạo dữ liệu synthetic để test...")
    
    # Tạo 1000 ảnh giả 32x32x3
    X = np.random.rand(1000, 32, 32, 3).astype(np.float32)
    # Tạo labels cho 43 classes
    y = np.random.randint(0, 43, 1000)
    y = to_categorical(y, 43)
    
    return X, y

def create_model():
    """Tạo model CNN"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(43, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_and_save_model():
    """Train và lưu model với dữ liệu synthetic"""
    try:
        # Thử load dữ liệu thật trước
        X, y = load_and_preprocess_data("../input/gtsrb-german-traffic-sign", "../input/gtsrb-german-traffic-sign/Train.csv")
        print("✅ Load dữ liệu thật thành công!")
    except:
        # Nếu không có dữ liệu thật, tạo dữ liệu synthetic
        print("⚠️ Không tìm thấy dataset, sử dụng synthetic data")
        X, y = create_synthetic_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.15, 
                               width_shift_range=0.1, height_shift_range=0.1)
    
    # Create and train model
    model = create_model()
    
    print("🔄 Đang train model...")
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                       validation_data=(X_test, y_test),
                       epochs=5)  # Giảm epochs để test nhanh
    
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save('models/traffic_model.h5')
    print("✅ Model đã được lưu tại 'models/traffic_model.h5'!")
    
    # Đánh giá model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"📊 Test Accuracy: {accuracy*100:.2f}%")
    
    return history

def load_and_preprocess_data(data_dir, train_csv_path):
    """Load và tiền xử lý dữ liệu thật"""
    train_csv = pd.read_csv(train_csv_path)
    
    X, y = [], []
    for i, row in train_csv.iterrows():
        img_path = os.path.join(data_dir, row['Path'])
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (32, 32))
            X.append(image)
            y.append(row['ClassId'])
    
    X = np.array(X) / 255.0
    y = to_categorical(y, 43)
    
    return X, y

if __name__ == "__main__":
    train_and_save_model()
