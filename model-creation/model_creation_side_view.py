# multi-class classification with Keras
import os
import random

import numpy as np
import pandas
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv(os.path.join('model-creation','datasets', 'dataset_side_view.csv'), header=0)
dataset = dataframe.values
dataset = dataset.tolist()


kyphotic_lordotic = dataset[0:50]
kyphotic = dataset[50:100]
lordotic = dataset[100:150]
normal = dataset[150:200]

for i in range(0, 50):
    new_el = []
    new_el.append(180.0-float(str(normal[i][0])))
    new_el.append(180.0-float(str(normal[i][1])))
    new_el.append(180.0-float(str(normal[i][2])))
    new_el.append(180.0-float(str(normal[i][3])))
    new_el.append('neutral-posture')
    normal.append(new_el)


shuffled = []
for i in range(0, 50):
    shuffled.append(normal[i])
    shuffled.append(kyphotic_lordotic[i])
    shuffled.append(kyphotic[i])
    shuffled.append(lordotic[i])
    shuffled.append(normal[50+i])

dataset = shuffled

# random.shuffle(dataset)

X = [i[:-1] for i in dataset]
Y = [i[-1] for i in dataset]

# for i in range(0, len(X)):
#     for y in range(0,3):
#         X[i][y] = round(X[i][y], 2)

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
model.fit(np.array(X), np.array(dummy_y), epochs=30, batch_size=10)
# Final evaluation of the model
scores = model.evaluate(np.array(X), np.array(dummy_y), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

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
