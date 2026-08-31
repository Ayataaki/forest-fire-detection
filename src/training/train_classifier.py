import mlflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.cnn_classifier import build_fire_cnn

def train(data_dir='../data/processed', epochs=50, batch_size=32):
    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=30, width_shift_range=0.2,
        height_shift_range=0.2, horizontal_flip=True,
        zoom_range=0.2, validation_split=0.2
    )
    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(224,224), batch_size=batch_size,
        subset='training', class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(224,224), batch_size=batch_size,
        subset='validation', class_mode='categorical'
    )
    model = build_fire_cnn()
    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        history = model.fit(train_gen, validation_data=val_gen,
                            epochs=epochs,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(patience=10),
                                tf.keras.callbacks.ModelCheckpoint(
                                    '../../models/fire_cnn_best.h5', save_best_only=True)
                            ])
        mlflow.log_metric('val_accuracy', max(history.history['val_accuracy']))
    return model
