####################### IMPORT MODULES #######################
import pandas
import torch
import numpy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



def data_preprocessing(task_1a_dataframe):
    # Create a copy of the original DataFrame to avoid modifying the original data
    # global encoded_dataframe
    encoded_dataframe = task_1a_dataframe.copy()
    # Normalize your input data if it's not already normalized

    # Initialize a LabelEncoder for encoding textual features
    label_encoder = LabelEncoder()

    # Encode the "Education" column
    encoded_dataframe['Education'] = label_encoder.fit_transform(encoded_dataframe['Education'])
    encoded_dataframe['JoiningYear'] = label_encoder.fit_transform(encoded_dataframe['JoiningYear'])
    encoded_dataframe['PaymentTier'] = label_encoder.fit_transform(encoded_dataframe['PaymentTier'])
    encoded_dataframe['Age'] = label_encoder.fit_transform(encoded_dataframe['Age'])
    # encoded_dataframe['ExperienceInCurrentDomain'] = label_encoder.fit_transform(encoded_dataframe['ExperienceInCurrentDomain'])
    # Encode the "City" column
    encoded_dataframe['City'] = label_encoder.fit_transform(encoded_dataframe['City'])

    # Encode the "Gender" column
    encoded_dataframe['Gender'] = label_encoder.fit_transform(encoded_dataframe['Gender'])

    # Encode the "EverBenched" column if it contains textual values
    if encoded_dataframe['EverBenched'].dtype == 'object':
        encoded_dataframe['EverBenched'] = label_encoder.fit_transform(encoded_dataframe['EverBenched'])


    return encoded_dataframe



def identify_features_and_targets(encoded_dataframe):

    # drop the target (leaveOrNot)
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])

    # store in target = 'LeaveOrNot'
    target = encoded_dataframe['LeaveOrNot']

    # Create a list containing the features and target label
    features_and_targets = [features, target]

    ##########################################################

    return features_and_targets


def load_as_tensors(features_and_targets):

    # Extract features and target from the input list
    features, target = features_and_targets

    # Split the data into training and validation sets (adjust the ratio as needed)
    train_ratio = 0.8  # 80% for training, 20% for validation
    num_samples = len(features)
    num_train_samples = int(train_ratio * num_samples)

    X_train = features[:num_train_samples]
    y_train = target[:num_train_samples]

    X_val = features[num_train_samples:]
    y_val = target[num_train_samples:]

    # Convert the training and validation data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)  # Reshape target to [batch_size, 1]
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)  # Reshape target to [batch_size, 1]

    # Create a TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Create DataLoader for training data to make it iterable in batches
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader]

    return tensors_and_iterable_training_data


class Salary_Predictor(nn.Module):
    # input_dim = 4634 
    
    def __init__(self, input_dim):
        super(Salary_Predictor, self).__init__()

        self.fc1 = nn.Linear(len(encoded_dataframe.columns) -1, 512)
        self.relu1 = nn.ReLU()               # Activation function (ReLU)
        self.fc2 = nn.Linear(512, 256)        # Hidden layer
        self.relu2 = nn.ReLU()               # Activation function (ReLU)
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()               # Activation function (ReLU)
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(32, 1)         # Output layer
       
        
      
    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        
        return torch.sigmoid(x)  # Apply sigmoid activation for binary classification




def model_loss_function():

    # Define the loss function for binary classification (cross-entropy loss)
    loss_function = nn.BCELoss()  # BCELoss stands for Binary Cross-Entropy Loss
    # loss_function = nn.MSELoss()
    ##########################################################

    return loss_function



def model_optimizer(model):

    # Define the optimizer
    learning_rate = 0.001  # You can adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ##########################################################

    return optimizer


def model_number_of_epochs():

    # Define the number of epochs
    number_of_epochs = 20  # we can adjust this value based on your experimentation

    ##########################################################

    return number_of_epochs



def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):


    # Extract training data and DataLoader
    X_train_tensor, _, y_train_tensor, _, train_loader = tensors_and_iterable_training_data

    # Training loop
    for epoch in range(number_of_epochs):
        model.train()  # Set the model to training mode

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print training loss for each epoch (optional)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}] - Loss: {loss.item()}")

    return model


def validation_function(trained_model, tensors_and_iterable_training_data):

    # Extract validation data and DataLoader
    _, X_val_tensor, _, y_val_tensor, _ = tensors_and_iterable_training_data

    # Set the model to evaluation mode
    trained_model.eval()

    # Disable gradient computation during validation
    with torch.no_grad():
        # Forward pass on the validation data
        outputs = trained_model(X_val_tensor)
        
        # Convert predicted probabilities to binary predictions (0 or 1)
        predicted_labels = (outputs >= 0.5).float()
        
        # Calculate accuracy
        correct_predictions = (predicted_labels == y_val_tensor).sum().item()
        total_samples = y_val_tensor.size(0)
        model_accuracy = correct_predictions / total_samples

    ##########################################################

    return model_accuracy



if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)


	model = Salary_Predictor(9)


	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
