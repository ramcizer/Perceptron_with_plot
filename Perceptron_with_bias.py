import numpy as np
import matplotlib.pyplot as plt


# Perceptron set as a class / object
class Perceptron:

    # Constructor initiates the classes attributes
    def __init__(self, num_features):
        self.num_features = num_features
        self.weight = np.zeros(num_features)       # One weight per feature
        self.bias = 0.0                             # Bias term (set to 0)
        self.weights_evo = []                       # List to store weight evolution                       


    # The formula that sets a potential update of weights
    def y_pred(self, x, y_label): 
        assert x.shape == self.weight.shape        # To avoid an error concerning vector dimensions
        prediction = y_label * (np.dot(self.weight, x) + self.bias)
        return prediction

    # The update if necessary
    def update(self, x, y_label):
        prediction = self.y_pred(x, y_label)

        if prediction <=  0:
            self.weight += y_label * x
            self.bias += y_label
            return 1
        return 0

# The function that sets the alorithm going
def train(model, all_x, all_y, max_epochs, convergence_threshold=1e-3):
    epoch_errors = []                               # List to store error counts per epoch

    # Loop by epoch, essential to understand convergence
    for epoch in range(max_epochs):
        error_count = 0   

        weight = model.weight 
        bias = model.bias        

        for x, y in zip(all_x, all_y):
            error = model.update(x, y)
            model.weights_evo.append(weight.copy())   #Weight append requires copy function
            error_count += abs(error)
        
        print(f'Epoch {epoch+1} errors {error_count}')
        
        epoch_errors.append(error_count)                # Append the error count

        if error_count < convergence_threshold:
            print(f'\nConverged after {epoch+1} epochs')
            break

    list_of_weights = [list(x) for x in model.weights_evo]

    list_of_weights = list(map(np.array, dict.fromkeys(map(tuple, list_of_weights)).keys()))

    print(f'\nThe total errors over epochs is {np.sum(epoch_errors)}.')  # Return error counts per epoch as a list
    print(f'The final bias is {bias}')
    print(f'\nA list of weight vector at the start of each epoch {list_of_weights}.\n')  # Print weight evolution

    return epoch_errors, list_of_weights, weight, bias


# Use Example, enter data for x and y labels: 

X_train = np.array([[-1, 1], [1, -1], [1, 1], [2, 2]])
y_train = np.array([1, 1, -1, -1])

ppn = Perceptron(num_features=2) # Update num_features based on vector shape
# Train the data and return the weight and bias values, remember to set epochs. 
epoch_errors, list_of_weights, weight, bias = train(ppn, all_x=X_train, all_y=y_train, max_epochs=10)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (- weight[0] * x0_1 - bias) / weight[1]
x1_2 = (- weight[0] * x0_2 - bias) / weight[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "b")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()







    

