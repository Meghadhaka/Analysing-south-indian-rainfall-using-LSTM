import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['figure.figsize']=10,6
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("/content/sample_data"))
dataset=pd.read_csv("/content/sample_data/rainfall in india 1901-2015.csv",encoding = "ISO-8859-1")
dataset.dtypes
groups = dataset.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','NOV','DEC']
data=groups.get_group(('BIHAR'))
data.head()
data=data.melt(['YEAR']).reset_index()
data.head()
df= data[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
df.head()
df.columns=['INDEX','YEAR','Month','avg_rainfall']
df.head()
d={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df['Month']=df['Month'].map(d)
df.head(12)
cols=['avg_rainfall']
dataset=df[cols]
dataset.head()
series=dataset
series.head()
series.shape
pyplot.figure(figsize=(20,6))
pyplot.plot(series.values)
pyplot.show()
# Get the raw data values from the pandas data frame.
data_raw = series.values.astype("float32")

# We apply the MinMax scaler from sklearn
# to normalize data in the (0, 1) interval.
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

# Print a few values.
dataset[0:5]
# Using 60% of data for training, 40% for validation.
TRAIN_SIZE = 0.80

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " + str((len(train), len(test))))
# FIXME: This helper function should be rewritten using numpy's shift function. See below.
def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))
# Create test and training sets for one-step-ahead regression.
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)

# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)
def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()

    model.add(LSTM(2000,activation = 'tanh', recurrent_activation = 'hard_sigmoid', input_shape = (1, window_size)))
    model.add(Dropout(0.2))
    model.add(Dense(500))
    model.add(Dropout(0.4))
    model.add(Dense(500))
    model.add(Dropout(0.4))
    model.add(Dense(400))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = "mean_squared_error",
                  optimizer = "adam")
    model.fit(train_X,
              train_Y,
              epochs = 10,
              batch_size = 64,
              )

    return(model)

# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)
import math
def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)
# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict

# Create the plot.
plt.figure(figsize = (18, 8))
plt.plot(scaler.inverse_transform(dataset), label = "True value",color='red')
plt.plot(train_predict_plot, label = "Training set prediction",color='yellow')
plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Months")
plt.legend()
plt.show()
test_predict
train_predict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Placeholder data for demonstration purposes (replace with your actual data)
subdivision_data = {
    "COASTAL ANDHRA PRADESH": {
        "train_X": np.random.rand(100, 5),
        "train_Y": np.random.rand(100),
        "test_X": np.random.rand(30, 5),
        "test_Y": np.random.rand(30)
    },
    "TELANGANA": {
        "train_X": np.random.rand(120, 5),
        "train_Y": np.random.rand(120),
        "test_X": np.random.rand(40, 5),
        "test_Y": np.random.rand(40)
    },
    "TAMIL NADU": {
        "train_X": np.random.rand(90, 5),
        "train_Y": np.random.rand(90),
        "test_X": np.random.rand(25, 5),
        "test_Y": np.random.rand(25)
    },
    "COASTAL KARNATAKA": {
        "train_X": np.random.rand(110, 5),
        "train_Y": np.random.rand(110),
        "test_X": np.random.rand(35, 5),
        "test_Y": np.random.rand(35)
    },
    "KERALA": {
        "train_X": np.random.rand(80, 5),
        "train_Y": np.random.rand(80),
        "test_X": np.random.rand(20, 5),
        "test_Y": np.random.rand(20)
    },
    "MADHYA MAHARASHTRA": {
        "train_X": np.random.rand(95, 5),
        "train_Y": np.random.rand(95),
        "test_X": np.random.rand(27, 5),
        "test_Y": np.random.rand(27)
    },
    "MANIPUR": {
        "train_X": np.random.rand(105, 5),
        "train_Y": np.random.rand(105),
        "test_X": np.random.rand(30, 5),
        "test_Y": np.random.rand(30)
    },
}

train_X = np.random.rand(100, 5)  # Replace with your actual train_X data
train_Y = np.random.rand(100)     # Replace with your actual train_Y data
test_X = np.random.rand(30, 5)    # Replace with your actual test_X data
test_Y = np.random.rand(30)       # Replace with your actual test_Y data

# Reshape data for LSTM input (assuming input_dim is 1)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_X, train_Y, epochs=10, batch_size=32)

# Function to perform predictions and score
def predict_and_score(model, X, Y):
    scaler = MinMaxScaler()
    scaler.fit(Y.reshape(-1, 1))  # Fit the scaler on the Y data

    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return score, pred

# Lists to accumulate predictions and states
all_predictions = []
all_states = []

# Iterate through subdivision_data for predictions
for state, data in subdivision_data.items():
    test_X = data["test_X"]
    test_Y = data["test_Y"]

    # Reshape data for LSTM input (assuming input_dim is 1)
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Perform predictions for the state
    rmse_test, test_predict = predict_and_score(model, test_X, test_Y)
    print(f"{state}: Test data score: %.2f RMSE" % rmse_test)

    # Check if test_predict is not None before appending
    if test_predict is not None:
        all_predictions.append(test_predict)
        all_states.append(state)

# Create a bar chart for all states
plt.figure(figsize=(14, 8))
for i in range(len(all_predictions)):
    plt.bar(np.arange(len(all_predictions[i])) + i * 0.2, all_predictions[i].flatten(), width=0.2, label=all_states[i])

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Test Set Predictions for States')
plt.legend()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Placeholder data for demonstration purposes (replace with your actual data)
subdivision_data = {
    "COASTAL ANDHRA PRADESH": {
        "train_X": np.random.rand(100, 5),
        "train_Y": np.random.rand(100),
        "test_X": np.random.rand(30, 5),
        "test_Y": np.random.rand(30)
    },
    "TELANGANA": {
        "train_X": np.random.rand(120, 5),
        "train_Y": np.random.rand(120),
        "test_X": np.random.rand(40, 5),
        "test_Y": np.random.rand(40)
    },
    "TAMIL NADU": {
        "train_X": np.random.rand(90, 5),
        "train_Y": np.random.rand(90),
        "test_X": np.random.rand(25, 5),
        "test_Y": np.random.rand(25)
    },
    "COASTAL KARNATAKA": {
        "train_X": np.random.rand(110, 5),
        "train_Y": np.random.rand(110),
        "test_X": np.random.rand(35, 5),
        "test_Y": np.random.rand(35)
    },
    "KERALA": {
        "train_X": np.random.rand(80, 5),
        "train_Y": np.random.rand(80),
        "test_X": np.random.rand(20, 5),
        "test_Y": np.random.rand(20)
    },
    "MADHYA MAHARASHTRA": {
        "train_X": np.random.rand(95, 5),
        "train_Y": np.random.rand(95),
        "test_X": np.random.rand(27, 5),
        "test_Y": np.random.rand(27)
    },
    "MANIPUR": {
        "train_X": np.random.rand(105, 5),
        "train_Y": np.random.rand(105),
        "test_X": np.random.rand(30, 5),
        "test_Y": np.random.rand(30)
    },
}

train_X = np.random.rand(100, 5)  # Replace with your actual train_X data
train_Y = np.random.rand(100)     # Replace with your actual train_Y data
test_X = np.random.rand(30, 5)    # Replace with your actual test_X data
test_Y = np.random.rand(30)       # Replace with your actual test_Y data

# Reshape data for LSTM input (assuming input_dim is 1)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_X, train_Y, epochs=10, batch_size=32)

# Function to perform predictions and score
def predict_and_score(model, X, Y):
    scaler = MinMaxScaler()
    scaler.fit(Y.reshape(-1, 1))  # Fit the scaler on the Y data

    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return score, pred

# Lists to store actual and predicted data for each state
actual_values = []
predicted_values = []
all_states = []

# Iterate through subdivision_data for predictions
for state, data in subdivision_data.items():
    test_X = data["test_X"]
    test_Y = data["test_Y"]

    # Reshape data for LSTM input (assuming input_dim is 1)
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Perform predictions for the state
    rmse_test, test_predict = predict_and_score(model, test_X, test_Y)
    print(f"{state}: Test data score: %.2f RMSE" % rmse_test)

    # Check if test_predict is not None before appending
    if test_predict is not None:
        actual_values.append(test_Y[:len(test_predict)])  # Trim actual values to match predicted values length
        predicted_values.append(test_predict.flatten())
        all_states.append(state)

# Find the maximum length among the actual and predicted values
max_len = max(len(arr) for arr in actual_values + predicted_values)

# Extend the arrays to match the maximum length
actual_values = [np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan) for arr in actual_values]
predicted_values = [np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan) for arr in predicted_values]

# Create a bar chart for actual vs predicted data for all states
plt.figure(figsize=(14, 8))
bar_width = 0.4
index = np.arange(max_len)

for i in range(len(actual_values)):
    plt.bar(index + (i * bar_width), actual_values[i], bar_width, label=f"Actual - {all_states[i]}")
    plt.bar(index + (i * bar_width) + bar_width, predicted_values[i], bar_width, label=f"Predicted - {all_states[i]}")

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Actual vs Predicted Data for States')
plt.legend()
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
dataset = pd.read_csv('/content/sample_data/rainfall in india 1901-2015.csv')

# Subdivisions list
subdivisions = [
    'COASTAL ANDHRA PRADESH', 'TELANGANA', 'TAMIL NADU', 'COASTAL KARNATAKA',
    'KERALA', 'MADHYA MAHARASHTRA', 'MANIPUR'
]

# Empty lists to store statistics for each subdivision
mean_values = []
median_values = []
max_values = []
min_values = []

# Calculate statistics for each subdivision
for subdivision in subdivisions:
    # Filter data for the current subdivision
    subdivision_data = dataset[dataset['SUBDIVISION'] == subdivision]

    # Calculate mean, median, max, and min
    mean_val = subdivision_data['ANNUAL'].mean()
    median_val = subdivision_data['ANNUAL'].median()
    max_val = subdivision_data['ANNUAL'].max()
    min_val = subdivision_data['ANNUAL'].min()

    # Append values to lists
    mean_values.append(mean_val)
    median_values.append(median_val)
    max_values.append(max_val)
    min_values.append(min_val)

    # Print values
    print(f"{subdivision}: Mean - {mean_val}, Median - {median_val}, Max - {max_val}, Min - {min_val}")

# Create a DataFrame to display the statistics
statistics_df = pd.DataFrame({
    'Subdivision': subdivisions,
    'Mean': mean_values,
    'Median': median_values,
    'Max': max_values,
    'Min': min_values
})

# Plotting the statistics for each subdivision
plt.figure(figsize=(12, 6))

# Set width of bar
bar_width = 0.2

# Set positions of bar on X axis
r1 = range(len(statistics_df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting the bars
plt.bar(r1, statistics_df['Mean'], color='b', width=bar_width, edgecolor='grey', label='Mean')
plt.bar(r2, statistics_df['Median'], color='g', width=bar_width, edgecolor='grey', label='Median')
plt.bar(r3, statistics_df['Max'], color='r', width=bar_width, edgecolor='grey', label='Max')
plt.bar(r4, statistics_df['Min'], color='y', width=bar_width, edgecolor='grey', label='Min')

# Adding labels
plt.xlabel('Subdivisions', fontweight='bold')
plt.xticks([r + bar_width * 1.5 for r in range(len(statistics_df))], statistics_df['Subdivision'], rotation=45)
plt.ylabel('Values', fontweight='bold')
plt.title('Rainfall Statistics by Subdivision')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
dataset = pd.read_csv('/content/sample_data/rainfall in india 1901-2015.csv')

# Subdivisions list
subdivisions = [
     'COASTAL ANDHRA PRADESH', 'TELANGANA', 'TAMIL NADU', 'COASTAL KARNATAKA',
    'KERALA', 'MADHYA MAHARASHTRA', 'MANIPUR'
]

# Plot individual graphs for each subdivision
for subdivision in subdivisions:
    # Filter data for the current subdivision
    subdivision_data = dataset[dataset['SUBDIVISION'] == subdivision]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(subdivision_data['YEAR'], subdivision_data['ANNUAL'], marker='o')
    plt.title(f'Annual Rainfall for {subdivision}')
    plt.xlabel('Year')
    plt.ylabel('Annual Rainfall (mm)')
    plt.grid(True)
    plt.tight_layout()

    # Show or save the plot
    plt.show()  # Uncomment to display individual plots
    # plt.savefig(f'{subdivision}_rainfall.png')  # Uncomment to save individual plots
import pandas as pd

# Read the dataset
dataset = pd.read_csv('/content/sample_data/rainfall in india 1901-2015.csv')

# Calculate the average annual rainfall across all subdivisions
average_rainfall = dataset['ANNUAL'].mean()

print(f"The average annual rainfall across all subdivisions is: {average_rainfall:.2f} mm")
import pandas as pd

# Read the dataset
dataset = pd.read_csv('/content/sample_data/rainfall in india 1901-2015.csv')

# List of subdivisions
subdivisions = [
    'COASTAL ANDHRA PRADESH', 'TELANGANA', 'TAMIL NADU', 'COASTAL KARNATAKA',
    'KERALA', 'MADHYA MAHARASHTRA', 'MANIPUR'
]

# Calculate average annual rainfall for each subdivision
for subdivision in subdivisions:
    subdivision_data = dataset[dataset['SUBDIVISION'] == subdivision]
    average_rainfall = subdivision_data['ANNUAL'].mean()
    print(f"Average annual rainfall in {subdivision}: {average_rainfall:.2f} mm")
