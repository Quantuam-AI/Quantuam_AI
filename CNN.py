from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# 이미지 데이터 경로 설정
train_data_dir = 'C:/Users/SEC/OneDrive/바탕 화면/종합프로젝트/Quantuam_AI/trainData'
validation_data_dir = 'C:/Users/SEC/OneDrive/바탕 화면/종합프로젝트/Quantuam_AI/validationData'

for class_folder in os.listdir(train_data_dir):
    class_path = os.path.join(train_data_dir, class_folder)
    if os.path.isdir(class_path):
        print(f'Class: {class_folder}')
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            print(f'  Image: {img_path}')

# 이미지 데이터 로드 (증강된 데이터)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 데이터 로드
img_width, img_height = 150, 150
batch_size = 32
epochs = 15



# train_generator와 validation_generator 정의
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)


print(f'Training Samples: {train_generator.samples}')


# CNN 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련


history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 테스트 데이터에 대한 예측
y_pred = model.predict(validation_generator)
y_pred = (y_pred > 0.5).astype(int)  # 이진 분류일 경우, 임계값을 0.5로 설정

# 실제 레이블
y_true = validation_generator.classes

# 추가적인 성능 지표 계산
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# 출력
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_true, y_pred))

# 손실과 정확도 그래프로 표시
plt.figure(figsize=(12, 4))

# 훈련 및 검증 손실
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 훈련 및 검증 정확도
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()