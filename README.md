# BreastCancer
The project involves predicting whether breast cancer is benign or malignant using data analysis and machine learning. The process involves several steps:

Data Preparation and Exploration:

Reading the dataset using Pandas.
Renaming columns for clarity.

Data Preprocessing:

Encoding categorical labels (Malignant and Benign) to numerical values (1 and 0) using LabelEncoder.
Scaling features using MinMaxScaler to bring them to a similar range.
Splitting the dataset into training and testing sets.

Model Building:

Constructing a neural network model using Keras Sequential API.
Adding dropout layers to prevent overfitting.
Compiling the model with binary crossentropy loss function and Adam optimizer.

Model Training and Evaluation:

Training the model on the training data.
Evaluating model performance using validation data.
Saving the trained model for future use.

Deployment with Streamlit:

Building a Streamlit web application for breast cancer prediction.
Providing a user interface for users to upload images for prediction.
Displaying predictions and recommending hospitals based on predictions.

Keywords:

Model Used : Sequential Neural Network using Keras Sequential API

	Input Layer:

	The input layer receives the input data, which consists of features extracted from breast cancer data.

	Hidden Layers:

	Two hidden layers are included in the model:

	The first hidden layer has 32 neurons and uses the Rectified Linear Unit (ReLU) activation function.
	The second hidden layer has 1 neuron and uses the Sigmoid activation function.

	Output Layer:

	The output layer consists of 1 neuron, representing the prediction of whether the breast cancer is benign or malignant.
	It uses the Sigmoid activation function to squash the output to a value between 0 and 1, representing the probability of malignancy.

	Dropout Layers:

	Dropout layers are added before each hidden layer with a dropout rate of 0.5.
	Dropout is used as a regularization technique to prevent overfitting by randomly dropping a fraction of input units.
	
	Compilation:

	The model is compiled using binary crossentropy loss function and the Adam optimizer.
	Binary crossentropy is a suitable loss function for binary classification tasks.
	Adam optimizer is an efficient optimization algorithm commonly used in training neural networks.



Used Version:

	Python : Python 3.9.16
	Pandas==1.3.3
	matplotlib==3.4.3
	seaborn==0.11.2
	numpy==1.21.2
	scikit-learn==0.24.2
	keras==2.6.0
	streamlit==1.5.0
	scikit-image==0.18.3


Steps to run :

python breastcancer.py       (Only one time)


streamlit run app.py
