import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("data.csv")

print(df.describe().T)

print(df.isnull().sum())
#df = df.dropna()

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis':'Label'})
print(df.dtypes)

#Understand the data
sns.countplot(x="Label", data=df) #M - malignant   B - benign

####### Replace categorical values with numbers########
print("Distribution of data: ", df['Label'].value_counts())

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
print("Labels before encoding are: ", np.unique(y))

# Encoding categorical data from text (B and M) to integers (0 and 1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # M=1 and B=0
print("Labels after encoding are: ", np.unique(Y))

#Define x and normalize / scale values

#Define the independent variables. Drop label and ID, and normalize other data
X = df.drop(labels = ["Label", "id","Unnamed: 32"], axis=1)
print(X.describe().T) #Needs scaling

#Scale / normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)  #Scaled values

#Split data into train and test to verify accuracy after fitting the model.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=40)
print("Shape of training data is: ", X_train.shape)
print("Shape of testing data is: ", X_test.shape)

""" sequential neural network. Sequential neural networks are a type of neural network that consists of a series of layers, each of which performs a specific function. The layers in a sequential neural network are typically stacked on top of each other, and the output of one layer is fed into the next layer.

The model you have provided has two hidden layers and one output layer. The hidden layers are responsible for learning the complex patterns in the data, and the output layer is responsible for making the final prediction.

The first hidden layer has 16 neurons, and it uses the rectified linear unit (ReLU) activation function. The ReLU activation function is a type of activation function that is often used in neural networks because it is simple and effective.

The second hidden layer has one neuron, and it uses the sigmoid activation function. The sigmoid activation function is a type of activation function that squashes the output of the neuron to a value between 0 and 1. This is useful for classification tasks, where the output of the neural network represents the probability of a particular class.

The output layer has one neuron, and it uses the sigmoid activation function. The output of the neural network is interpreted as the probability of the input data being malignant.

The model is compiled using the binary crossentropy loss function and the Adam optimizer. The binary crossentropy loss function is a type of loss function that is often used for classification tasks. The Adam optimizer is a type of optimizer that is often used for training neural networks because it is efficient and effective.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dropout(0.5)) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.5))  
model.add(Dense(1, activation='sigmoid')) 
model.build(input_shape=(None, 30))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#Fit with no early stopping or other callbacks
history = model.fit(X_train, y_train, verbose=1, epochs=100, batch_size=64,
                    validation_data=(X_test, y_test))

model.save("breast_cancer_model.h5")

#plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# acc = history.history['accuracy']  #Use accuracy if acc doesn't work
# val_acc = history.history['val_accuracy']  #Use val_accuracy if acc doesn't work
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Predicting the Test set results
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# sns.heatmap(cm, annot=True)