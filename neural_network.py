import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Load the dataset
df = pd.read_csv('Car_Purchasing_Data.csv')

# Create input dataset from original dataset by dropping irrelevant features
X = df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
Y = df['Car Purchase Amount']

# Transform input dataset into percentage-based weighted between 0 and 1
sc = MinMaxScaler()
x_scaled = sc.fit_transform(X)

# Transform output dataset into percentage-based weighted between 0 and 1
sc1 = MinMaxScaler()
y_reshape = Y.values.reshape(-1, 1)
y_scaled = sc1.fit_transform(y_reshape).ravel()  # Convert to 1D array

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#Define a Simple Neural Network using PyTorch
#It consists of three fully connected layers (Linear), with ReLU activation functions for the first two layers and no activation for the output layer (self.layer3).
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(5, 64)  # Input layer
        self.layer2 = nn.Linear(64, 32)  # Hidden layer
        self.layer3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Initialize own model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the other models
linear_model = LinearRegression()
svm = SVR()
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
ridge = Ridge()
en = ElasticNet()
rr = RANSACRegressor()
dtr = DecisionTreeRegressor()
ann = MLPRegressor(max_iter=1000)
etr = ExtraTreesRegressor()

# Training own model
num_epochs = 1000
losses = []  # Initialize the list to store loss values
#Training Loop: Trains the neural network for num_epochs epochs. It calculates predictions (outputs), computes loss against the training target (y_train_tensor), performs backpropagation (loss.backward()), and updates the model weights (optimizer.step()).
#Loss Tracking: Stores the loss values (loss.item()) in the losses list for plotting later.
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# less training loss = more accurate

# Train the other models using the training set
linear_model.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
en.fit(X_train, y_train)
rr.fit(X_train, y_train)
dtr.fit(X_train, y_train)
ann.fit(X_train, y_train)
etr.fit(X_train, y_train)

# Plotting the loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Prediction on the validation/test data
linear_model_preds = linear_model.predict(X_test)
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)
gbr_preds = gbr.predict(X_test)
ridge_preds = ridge.predict(X_test)
en_preds = en.predict(X_test)
rr_preds = rr.predict(X_test)
dtr_preds = dtr.predict(X_test)
ann_preds = ann.predict(X_test)
etr_preds = etr.predict(X_test)

# Inverse transform the predictions and the true values
y_test_original = sc1.inverse_transform(y_test.reshape(-1, 1))
linear_model_preds = sc1.inverse_transform(linear_model_preds.reshape(-1, 1))
svm_preds = sc1.inverse_transform(svm_preds.reshape(-1, 1))
rf_preds = sc1.inverse_transform(rf_preds.reshape(-1, 1))
gbr_preds = sc1.inverse_transform(gbr_preds.reshape(-1, 1))
ridge_preds = sc1.inverse_transform(ridge_preds.reshape(-1, 1))
en_preds = sc1.inverse_transform(en_preds.reshape(-1, 1))
rr_preds = sc1.inverse_transform(rr_preds.reshape(-1, 1))
dtr_preds = sc1.inverse_transform(dtr_preds.reshape(-1, 1))
ann_preds = sc1.inverse_transform(ann_preds.reshape(-1, 1))
etr_preds = sc1.inverse_transform(etr_preds.reshape(-1, 1))

# Evaluating the PyTorch model
model.eval() #Sets the model to evaluation mode. 
with torch.no_grad(): #Context manager that ensures no gradients are calculated within this block.
    predictions = model(X_test_tensor)
    predictions = sc1.inverse_transform(predictions.numpy())
    y_test_original = sc1.inverse_transform(y_test_tensor.numpy())
    mse = np.mean((predictions - y_test_original) ** 2)
    sn_rmse = np.sqrt(mse)
    print(f'Simple Neural Network RMSE: {sn_rmse:.4f}')

# Evaluate the other model performance
linear_model_rmse = mean_squared_error(y_test_original, linear_model_preds, squared=False)
svm_rmse = mean_squared_error(y_test_original, svm_preds, squared=False)
rf_rmse = mean_squared_error(y_test_original, rf_preds, squared=False)
gbr_rmse = mean_squared_error(y_test_original, gbr_preds, squared=False)
ridge_rmse = mean_squared_error(y_test_original, ridge_preds, squared=False)
en_rmse = mean_squared_error(y_test_original, en_preds, squared=False)
rr_rmse = mean_squared_error(y_test_original, rr_preds, squared=False)
dtr_rmse = mean_squared_error(y_test_original, dtr_preds, squared=False)
ann_rmse = mean_squared_error(y_test_original, ann_preds, squared=False)
etr_rmse = mean_squared_error(y_test_original, etr_preds, squared=False)

# Display the evaluation results
print(f"Linear Regression RMSE: {linear_model_rmse}")
print(f"Support Vector Machine RMSE: {svm_rmse}")
print(f"Random Forest Regressor RMSE: {rf_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"Ridge Regression RMSE: {ridge_rmse}")
print(f"Elastic Net RMSE: {en_rmse}")
print(f"Robust Regression RMSE: {rr_rmse}")
print(f"Decision Tree Regressor RMSE: {dtr_rmse}")
print(f"Artificial Neural Network RMSE: {ann_rmse}")
print(f"Extra Trees Regressor RMSE: {etr_rmse}")

# Choose the best model
model_objects = [linear_model, svm, rf, gbr, ridge, en, rr, dtr, ann, etr]
rmse_values = [linear_model_rmse, svm_rmse, rf_rmse, gbr_rmse, ridge_rmse, en_rmse, rr_rmse, dtr_rmse, ann_rmse, etr_rmse]

# Add Simple Neural Network RMSE to the list
model_objects.append(model)
rmse_values.append(sn_rmse)

# Visualize the model results
models = ['Linear Regression', 'Support Vector Machine', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'Ridge Regression', 'Elastic Net', 'Robust Regression', 'Decision Tree Regressor', 'Artificial Neural Network', 'Extra Trees Regressor', 'Simple Neural Network']

plt.figure(figsize=(12, 6))
bars = plt.bar(models, rmse_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()

# Add RMSE values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom', fontsize=10)

plt.show()

# Save the PyTorch model
torch.save(model.state_dict(), 'simple_nn_model.pth')

# Function to make prediction with user
def predict_car_purchase_amount(gender, age, annual_salary, credit_card_debt, net_worth):
    input_data = np.array([[gender, age, annual_salary, credit_card_debt, net_worth]])
    input_data_scaled = sc.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_amount = sc1.inverse_transform(prediction.numpy())
        return predicted_amount[0][0]

try:
    gender = float(input("Enter gender (0 for female, 1 for male): "))
    age = float(input("Enter age: "))
    annual_salary = float(input("Enter annual salary: "))
    credit_card_debt = float(input("Enter credit card debt: "))
    net_worth = float(input("Enter net worth: "))
    predicted_amount = predict_car_purchase_amount(gender, age, annual_salary, credit_card_debt, net_worth)
    print(f"Predicted Car Purchase Amount based on input: {predicted_amount}")
except ValueError as e:
    print(f"Invalid input: {e}")
    exit()