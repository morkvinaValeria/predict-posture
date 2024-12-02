from keras.models import load_model
import pandas
import os

# Work with the trained model
# Loading the trained model
model = load_model(os.path.join('trained-model',
                                'posture_side_view_assessment_trained_model.h5'))
# Loading data
validate_dataframe = pandas.read_csv(os.path.join('test-predict','datasets', 'test_side_view.csv'), header=0)
validate_dataset = validate_dataframe.values
val_X = [i[:-1] for i in validate_dataset.tolist()]
val_Y = [i[-1] for i in validate_dataset.tolist()]

labels = {'kyphotic-lordotic-posture': 0, 'kyphotic-posture': 1, 'lordotic-posture': 2, 'neutral-posture': 3}
preds = model.predict(val_X)
preds = preds.tolist()
# print results
print(labels)
for i in range(0, len(preds)):
    print(
        f"\nitem#{i}: {max(preds[i])} - {list(labels.keys())[list(labels.values()).index(preds[i].index(max(preds[i])))]}, but expected - {val_Y[i]}")
    print(preds[i])
