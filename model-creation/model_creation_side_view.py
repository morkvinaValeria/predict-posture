# multi-class classification with Keras
import os
import random

import numpy as np
import pandas
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv(os.path.join('model-creation','datasets', 'dataset_side_view.csv'), header=0)
dataset = dataframe.values
dataset = dataset.tolist()


kyphotic_lordotic = dataset[0:50]
kyphotic = dataset[50:100]
lordotic = dataset[100:150]
normal = dataset[150:250]

val_shuffled = []
for i in range(0, 5):
    val_shuffled.append(normal[i])
    val_shuffled.append(kyphotic_lordotic[i])
    val_shuffled.append(kyphotic[i])
    val_shuffled.append(lordotic[i])
    val_shuffled.append(normal[5+i])

shuffled = []
for i in range(0, 50):
    shuffled.append(normal[i])
    shuffled.append(kyphotic_lordotic[i])
    shuffled.append(kyphotic[i])
    shuffled.append(lordotic[i])
    shuffled.append(normal[50+i])

dataset = shuffled

X = [i[:-1] for i in dataset]
Y = [i[-1] for i in dataset]

X_val = [i[:-1] for i in val_shuffled]
Y_val = [i[-1] for i in val_shuffled]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)
model = Sequential([
                    Dense(350, input_dim=4, activation='relu'),
                    Dense(250, activation='relu'),
                    Dropout(0.3),
                    Dense(200, activation='relu'),
                    Dropout(0.2),
                    Dense(150, activation='relu'),
                    Dropout(0.1),
                    Dense(90, activation='relu'),
                    Dropout(0.1),
                    Dense(40, activation='relu'),
                    Dense(4, activation='softmax')])
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history= model.fit(np.array(X), np.array(dummy_y), epochs=30, batch_size=10, validation_data=(np.array(X_val), np.array(to_categorical(encoder.transform(Y_val)))))
# Final evaluation of the model
scores = model.evaluate(np.array(X), np.array(dummy_y), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Validate model
validate_dataframe = pandas.read_csv("test_side_view.csv", header=0)
validate_dataset = validate_dataframe.values
val_X = [i[:-1] for i in validate_dataset.tolist()]
val_Y = [i[-1] for i in validate_dataset.tolist()]

labels = dict(zip(encoder.classes_, range(len(encoder.classes_))))
print(labels)
preds = model.predict(val_X)
preds = preds.tolist()
for i in range(0, len(preds)):
    print(
        f"\nitem#{i}: {max(preds[i])} - {list(labels.keys())[list(labels.values()).index(preds[i].index(max(preds[i])))]}, but expected - {val_Y[i]}")
    print(preds[i])


# save the trained model
model.save(os.path.join('trained-model',
           'posture_side_view_assessment_trained_model3.h5'))
