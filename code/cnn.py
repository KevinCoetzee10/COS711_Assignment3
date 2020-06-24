import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import math
from keras import backend as K

# helper functions
def cleanMissingValues(dataset):
    print('cleaning dataset.')
    cleanedDataset = dataset
    shape = cleanedDataset.shape
    if shape[1] == 8:
        cleanedDataset["target"] = 0
    cleanedDataset["missing"] = False
    rowDroppedCount = 0
    for x in range(0, shape[0]):
        maxRowMissingValuesReached = False
        rowDropThreshold = len(cleanedDataset.iloc[x, 2]) / 10 * 9
        for y in range(2, 8):
            nanCount = 0
            meanTotal = 0
            meanNum = 0
            maxColMissingValuesReached = False
            nanPresent = False
            for z in cleanedDataset.iloc[x, y]:
                if math.isnan(z):
                    nanCount = nanCount + 1
                    nanPresent = True
                elif maxRowMissingValuesReached == False:
                    meanNum = meanNum + 1
                    meanTotal = meanTotal + z
            if nanCount > rowDropThreshold:
                maxRowMissingValuesReached = True
                maxColMissingValuesReached = True
            elif nanPresent and maxRowMissingValuesReached == False:
                mean = meanTotal / meanNum
                for z in range(0, len(cleanedDataset.iloc[x, y])):
                    if math.isnan(cleanedDataset.iloc[x, y][z]):
                        cleanedDataset.iloc[x, y][z] = mean
        if maxRowMissingValuesReached:
            cleanedDataset.iloc[x, 9] = True
    cleanedDataset = cleanedDataset[cleanedDataset.missing == False]
    cleanedDataset = cleanedDataset.drop("missing", axis=1)
    shape = cleanedDataset.shape
    return cleanedDataset

def reshapeDataSet(datasetTrain, datasetValidate):
    print('reshaping dataset.')
    expandedDatasetTrain = datasetTrain
    expandedDatasetValidate = datasetValidate
    shape = expandedDatasetTrain.shape
    cubeTrain = list()
    for x in range(0, shape[0]):
        square = list()
        for y in range(0, 6):
            column = list()
            for z in expandedDatasetTrain.iloc[x, y]:
                column.append(z)
            square.append(column)
        cubeTrain.append(square)
    cubeTrain = np.dstack(cubeTrain)
    shape = expandedDatasetValidate.shape
    cubeValidate = list()
    for x in range(0, shape[0]):
        square = list()
        for y in range(0, 6):
            column = list()
            for z in expandedDatasetValidate.iloc[x, y]:
                column.append(z)
            square.append(column)
        cubeValidate.append(square)
    cubeValidate = np.dstack(cubeValidate)
    for x in range(0, cubeTrain.shape[0]):
        sliceToNormaliseTrain = cubeTrain[x, :, :]
        sliceToNormaliseValidate = cubeValidate[x, :, :]
        sliceToNormaliseTrainMax = np.max(sliceToNormaliseTrain)
        sliceToNormaliseValidateMax = np.max(sliceToNormaliseValidate)
        sliceToNormaliseTrainMin = np.min(sliceToNormaliseTrain)
        sliceToNormaliseValidateMin = np.min(sliceToNormaliseValidate)
        maxToUse = 0
        minToUse = 0
        if sliceToNormaliseTrainMax > sliceToNormaliseValidateMax:
            maxToUse = sliceToNormaliseTrainMax
        else:
            maxToUse = sliceToNormaliseValidateMax
        if sliceToNormaliseTrainMin < sliceToNormaliseValidateMin:
            minToUse = sliceToNormaliseTrainMin
        else:
            minToUse = sliceToNormaliseValidateMin
        sliceToNormaliseTrain = (sliceToNormaliseTrain - minToUse) / (maxToUse - minToUse)
        sliceToNormaliseValidate = (sliceToNormaliseValidate - minToUse) / (maxToUse - minToUse)
        cubeTrain[x, :, :] = sliceToNormaliseTrain
        cubeValidate[x, :, :] = sliceToNormaliseValidate
    return cubeTrain, cubeValidate

# reading in necessary train and test files
trainDataUrl = '../data/Train.p'
testDataUrl = '../data/Test.p'
infile = open(trainDataUrl,'rb')
trainData = pickle.load(infile)
infile.close()
infile = open(testDataUrl,'rb')
testData = pickle.load(infile)
infile.close()

# # reading in pre-processed data
# trainDataUrl = 'shapableTraining.pickle'
# # testDataUrl = '../data/Test.p'
# infile = open(trainDataUrl,'rb')
# trainData = pickle.load(infile)
# infile.close()
# # infile = open(testDataUrl,'rb')
# # testData = pickle.load(infile)
# # infile.close()

print("dataset samples before cleaning:")
print(trainData.shape[0])

# cleaning missing input values
trainData = cleanMissingValues(trainData)

# splitting X and Y values
Y_train = trainData.target
X_train = trainData.drop('target', axis=1)
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, train_size=0.8, random_state=50)
print("training set samples:")
print(X_train.shape[0])
print("validation set samples:")
print(X_validate.shape[0])

# reshaping data for CNN
X_train = X_train.iloc[:, 2:8]
X_validate = X_validate.iloc[:, 2:8]
X_train, X_validate = reshapeDataSet(X_train, X_validate)

# saving shaped dataset
pickle_out = open("shapableTraining.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

X_train_CNN = np.reshape(X_train, (X_train.shape[2], 6, 121, 1))
X_validate_CNN = np.reshape(X_validate, (X_validate.shape[2], 6, 121, 1))

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights("model.h5")
# print("Loaded model from disk")

# build CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (6, 121, 1), border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))
model.add(Dropout(0.05))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))
model.add(Dropout(0.15))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
opt = adam(lr=0.0001)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print('training model.')
model.fit(x=X_train_CNN, y=Y_train, validation_data=(X_validate_CNN, Y_validate), epochs=1000, batch_size=16, callbacks=[es])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# making predictions
print('making predictions.')
preds = model.predict(X_validate_CNN)

diff = preds.flatten() - Y_validate
percentDiff = (diff / Y_validate) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print('mean: {:.2f}%, std: {:.2f}%'.format(mean, std))