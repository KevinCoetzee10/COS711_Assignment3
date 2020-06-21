import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import math

# helper functions
def cleanMissingValues(dataset):
    print('cleaning dataset.')
    cleanedDataset = dataset
    shape = cleanedDataset.shape
    if shape[1] == 8:
        cleanedDataset["target"] = 0
    cleanedDataset["missing"] = False
    rowDroppedCount = 0
    # print("rows before clean: ")
    # print(shape[0])
    for x in range(0, shape[0]):
        maxRowMissingValuesReached = False
        rowDropThreshold = len(cleanedDataset.iloc[x, 2]) / 3
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
    if shape[1] == 8:
        cleanedDataset = cleanedDataset.drop("target", axis=1)
    shape = cleanedDataset.shape
    # print("rows after clean: ")
    # print(shape[0])
    return cleanedDataset

def reshapeDataSet(dataset):
    print('reshaping dataset.')
    expandedDataset = dataset
    shape = expandedDataset.shape
    cube = list()
    for x in range(0, shape[0]):
        # print(x)
        square = list()
        for y in range(0, 6):
            # for z in range(0, len(expandedDataset.iloc[x, y])):
            column = list()
            for z in expandedDataset.iloc[x, y]:
                column.append(z)
                # label = str(expandedDataset.index[y]) + str(z)
                # expandedDataset[label] = expandedDataset.iloc[x, y][z]
            square.append(column)
        cube.append(square)
    cube = np.dstack(cube)

    # expandedDataset = expandedDataset.drop('temp', axis=1)
    # expandedDataset = expandedDataset.drop('precip', axis=1)
    # expandedDataset = expandedDataset.drop('rel_humidity', axis=1)
    # expandedDataset = expandedDataset.drop('wind_dir', axis=1)
    # expandedDataset = expandedDataset.drop('wind_spd', axis=1)
    # expandedDataset = expandedDataset.drop('atmos_press', axis=1)
    return cube

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

# combined train and test data sets for data preparation
# trainData['type'] = 'train'
# testData['type'] = 'test'
# fullData = pd.concat([trainData, testData], axis=0)

# cleaning missing input values
trainData = cleanMissingValues(trainData)
# testData = cleanMissingValues(testData)

# fullData = cleanMissingValues(fullData)

# print(trainData[trainData.ID == "ID_train_10011"])

# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                     train_size=0.67, 
#                                                     random_state=42)

# splitting X and Y values
Y_train = trainData.target
X_train = trainData.drop('target', axis=1)
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, train_size=0.8, random_state=50)
print("training set samples:")
print(X_train.shape[0])
print("validation set samples:")
print(X_validate.shape[0])
# Y_test = testData.target
# X_test = testData.drop('target', axis=1)

# reshaping data for CNN
X_train = X_train.iloc[:, 2:8]
X_train = reshapeDataSet(X_train)
X_validate = X_validate.iloc[:, 2:8]
X_validate = reshapeDataSet(X_validate)
# numpyArray = list()
# numpyArray.append(X_train)
# numpyArray.dstack(numpyArray)

pickle_out = open("shapableTraining.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

X_train_CNN = np.reshape(X_train, (10920, 6, 121, 1))
X_validate_CNN = np.reshape(X_validate, (2731, 6, 121, 1))
# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

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
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', input_shape = (6, 121, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# opt = adam(lr=1e-3, decay=1e-3 / 200)
opt = adam(lr=1e-2)
# model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['accuracy'])
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

print('training model.')
model.fit(x=X_train_CNN, y=Y_train, validation_data=(X_validate_CNN, Y_validate), epochs=200, batch_size=8, callbacks=[es])
# model.fit(x=X_train_CNN, y=Y_train, validation_data=(X_validate_CNN, Y_validate), epochs=200, batch_size=8)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print('making predictions.')
preds = model.predict(X_validate_CNN)

diff = preds.flatten() - Y_validate
percentDiff = (diff / Y_validate) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print('mean: {:.2f}%, std: {:.2f}%'.format(mean, std))