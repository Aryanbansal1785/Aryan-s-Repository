import numpy as np  #contains mathematical values
import pandas as pd #helps in accessing and reading files
import matplotlib.pyplot as plt #helps in graphical representation  
import yfinance as yf
import datetime as dt
import matplotlib.dates as mdates
 
 
from sklearn.preprocessing import MinMaxScaler #MinMaxScaler is a preprocessing technique commonly used in machine learning to scale and normalize the numerical features of a dataset.
from tensorflow.keras.models import Sequential  #n TensorFlow's Keras API, the Sequential class is a linear stack of layers, allowing for the creation of a model layer by layer in a step-by-step fashion.
from tensorflow.keras.layers import Dense, Dropout, LSTM  #LSTM - Long short term memory layers

#sklearn api provides efficient tools for data analysis and modeling and machine learning algorithms and data preproccessing.
#tensorflow api provide wide set of tools and libraries for building and deploying ML models.Provides wide range of functionalities for data pre-processing, training,evaluation

#Load data
company = 'AMZN' #AMZN is ticker symbol

start = dt.datetime(2023, 1, 1)
end = dt.datetime(2024, 1, 1)

data = yf.download(company, start=start, end=end )

#prepare data
scaler= MinMaxScaler(feature_range=(0,1))  #scaling all values so that they fit in 0 and 1 
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))  #here it means we are not transforming the whole dataframe but only the closing price

prediction_days = 60

x_train =[]
y_train =[]

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])  #this will be the 61st trained data
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape (x_train, (x_train.shape[0], x_train.shape[1], 1)) #reshape for neural networks

# Print or check the dimensions of x_train
print("x_train shape:", x_train.shape)

#build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of the next price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32) #fitting the model #batch size means that the model is going to see 32 batches at one time


'''Test the Model'''

#Load The Test

test_start= dt.datetime(2023,1,1)  #This data has to be the one that model hasn't seen before
test_end= dt.datetime.now()

test_data= yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat([data['Close'], test_data['Close']])
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Make predictions on Test Data

x_test=[]

for x in range(prediction_days, len(model_inputs) ):
    x_test.append(model_inputs[x-prediction_days:x,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1,1))


# Get the common indices
common_indices = test_data.index[-len(predicted_prices):]

# Plot the test predictions
if len(common_indices) > 0:
    # Convert common_indices to a list
    common_indices_list = list(common_indices)

    # Trim predicted_prices to match the length of common_indices
    predicted_prices_trimmed = predicted_prices[-len(common_indices):]

    plt.plot(common_indices_list, actual_prices[-len(common_indices):], color="Red", label=f"Actual {company} Price")
    plt.plot(common_indices_list, predicted_prices_trimmed, color="Blue", label=f"Predicted {company} Price")

    # Format the x-axis to show dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()
else:
    print("No common indices available for plotting.")
