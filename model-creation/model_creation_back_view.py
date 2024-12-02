# multi-class classification with Keras
import os
import random

import numpy as np
import pandas
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("./datasets/dataset_back_view.csv", header=0)
dataset = dataframe.values
dataset = dataset.tolist()


normal = dataset[0:100]
right_c_sc = dataset[100:150]
left_c_sc = dataset[150:200]
s_sc = dataset[200:250]

# for i in range(0, 50):
#     new_el = []
#     new_el.append(180.0-float(str(right_c_sc[i][0])))
#     new_el.append(180.0-float(str(right_c_sc[i][1])))
#     new_el.append(180.0-float(str(right_c_sc[i][2])))
#     new_el.append('left-c-scoliotic-posture')
#     left_c_sc.append(new_el)

#     new_el_2 = []
#     new_el_2.append(180.0-float(str(left_c_sc[i][0])))
#     new_el_2.append(180.0-float(str(left_c_sc[i][1])))
#     new_el_2.append(180.0-float(str(left_c_sc[i][2])))
#     new_el_2.append('right-c-scoliotic-posture')
#     right_c_sc.append(new_el_2)

#     new_el_3 = []
#     new_el_3.append(180.0-float(str(s_sc[i][0])))
#     new_el_3.append(180.0-float(str(s_sc[i][1])))
#     new_el_3.append(180.0-float(str(s_sc[i][2])))
#     new_el_3.append('normal-case')
#     s_sc.append(new_el_3)

# for i in range(0, 100):
#     angle_1, angle_2, angle_3 = normal[i][0], normal[i][1], normal[i][2]
#     new_el = []
#     new_el.append(angle_3)
#     new_el.append(angle_1)
#     new_el.append(angle_2)
#     new_el.append('normal-case')
#     normal.append(new_el)


shuffled = []
for i in range(0, 50):
    shuffled.append(normal[i])
    shuffled.append(right_c_sc[i])
    shuffled.append(left_c_sc[i])
    shuffled.append(s_sc[i])
    shuffled.append(normal[50+i])
    # shuffled.append(normal[100+i])
    # shuffled.append(right_c_sc[50+i])
    # shuffled.append(left_c_sc[50+i])
    # shuffled.append(s_sc[50+i])
    # shuffled.append(normal[150+i])

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
                    Dense(300, input_dim=3, activation='relu'),
                    Dense(200, activation='relu'),
                    Dropout(0.3),
                    Dense(150, activation='relu'),
                    Dropout(0.2),
                    Dense(100, activation='relu'),
                    Dropout(0.1),
                    Dense(70, activation='relu'),
                    Dropout(0.1),
                    Dense(20, activation='relu'),
                    Dense(4, activation='softmax')])
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(np.array(X), np.array(dummy_y), epochs=25, batch_size=10)
# Final evaluation of the model
scores = model.evaluate(np.array(X), np.array(dummy_y), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# validate_dataframe = pandas.read_csv("test.csv", header=0)
# validate_dataset = validate_dataframe.values
# val_X = [i[:-1] for i in validate_dataset.tolist()]
# val_Y = [i[-1] for i in validate_dataset.tolist()]

# labels = dict(zip(encoder.classes_, range(len(encoder.classes_))))
# print(labels)
# preds = model.predict(val_X)
# preds = preds.tolist()
# for i in range(0, len(preds)):
#     print(
#         f"\nitem#{i}: {max(preds[i])} - {list(labels.keys())[list(labels.values()).index(preds[i].index(max(preds[i])))]}, but expected - {val_Y[i]}")
#     print(preds[i])


# save the trained model
model.save(os.path.join('trained-model',
           'posture_assessment_trained_model.h5'))
