from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 이미지 데이터 경로 설정
train_data_dir = 'Quantuam_AI/trainData'
validation_data_dir = 'Quantuam_AI/validationData'

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

# 모델 평가
score = model.evaluate(validation_generator)
print(f'Test Loss: {score[0]}, Test Accuracy: {score[1]}')

# 훈련 및 검증 손실, 정확도 출력
print("Training History:")
print(history.history)
