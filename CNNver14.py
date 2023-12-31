from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 이미지 데이터 경로 설정
train_data_dir = 'Quantuam_AI/trainData3'
validation_data_dir = 'Quantuam_AI/validationData'
test_data_dir = 'Quantuam_AI/testData'  # 테스트 데이터 경로

# 이미지 데이터 로드 (증강된 데이터)
test_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 데이터 로드
img_width, img_height = 150, 150
batch_size = 32
epochs = 20

# train_generator, validation_generator, test_generator 정의
train_generator = test_datagen.flow_from_directory(
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

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
# CNN 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

# 출력 뉴런 수를 1로 변경하고 활성화 함수를 sigmoid로 변경
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
optimizer = Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 모델 훈련
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# 테스트 데이터에 대한 실제 레이블
y_true = validation_generator.classes

# 테스트 데이터에 대한 예측
y_pred = model.predict(validation_generator)

# 이진 분류이므로 threshold를 설정하여 이진 분류로 변환
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)

# 추가적인 성능 지표 계산
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary, zero_division=0)
recall = recall_score(y_true, y_pred_binary, zero_division=0)
f1 = f1_score(y_true, y_pred_binary, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred_binary)

# 클래스 0에 대한 정밀도, 재현율, F1 점수 출력
precision_class_0 = precision_score(y_true, y_pred_binary, pos_label=0, zero_division=0)
recall_class_0 = recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0)
f1_class_0 = f1_score(y_true, y_pred_binary, pos_label=0, zero_division=0)

# 출력
print(f'Test Accuracy: {accuracy}')
print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')
print(f'Test F1 Score: {f1}')
print('Test Confusion Matrix:')
print(conf_matrix)

# 클래스 0에 대한 성능 지표 출력
print(f'Precision for Class 0: {precision_class_0}')
print(f'Recall for Class 0: {recall_class_0}')
print(f'F1 Score for Class 0: {f1_class_0}')

print('Test Classification Report:')
print(classification_report(y_true, y_pred_binary))

# 훈련 및 검증 손실 그래프로 표시
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
