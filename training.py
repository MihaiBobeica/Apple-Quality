import numpy as np
import pandas as pd


# sigmoid function to normalize inputs
def sigmoid(x):
    clipped_x = np.clip(x, -500, 500)  # Clip x to prevent overflow
    return 1 / (1 + np.exp(-clipped_x))


# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)


csv_file = 'apple_quality.csv'

# Load CSV file into a pandas DataFrame
data_frame = pd.read_csv(csv_file)

# Assuming the last column is the expected result (output)
training_outputs_array = data_frame.iloc[:, -1].to_numpy().T.reshape(-1,1)

# Assuming all other columns are input values
training_inputs_array = data_frame.iloc[:, 1:-1].to_numpy()



#print("Training Inputs:")
#print(training_inputs_array)
#print(np.array2string(training_inputs_array, threshold=np.inf, separator=', '))
#
#print("\nTraining Outputs:")
#print(training_outputs_array)
#print("Training outPuts shape:", training_outputs_array.shape)
#print(np.array2string(training_outputs_array, threshold=np.inf, separator=', '))


# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = np.random.random((7,1))
#print("\n Synaptic Weight:")
#print(synaptic_weights)
#print('Random starting synaptic weights: ')
#print(synaptic_weights)

# Iterate 10,000 times
for iteration in range(1000000):

    # Define input layer
    input_layer = training_inputs_array
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs_array - outputs

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T,adjustments)

print("weights After Training:")
print(synaptic_weights)