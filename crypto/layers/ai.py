import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Define the CryptoChartPredictor class using PyTorch LSTM model
class CryptoChartPredictor:
    def __init__(self, data: list, look_back=60):
        self.data = data
        self.look_back = look_back
        self.model = None

        # Convert the ChartData list into a Pandas DataFrame
        self.df = self._create_dataframe()

        # Preprocess data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X, self.y = self._prepare_data()

    def _create_dataframe(self):
        """
        Converts the list of ChartData into a pandas DataFrame.
        """
        data_dict = {
            'timestamp': [c.timestamp for c in self.data],
            'open': [c.open for c in self.data],
            'high': [c.high for c in self.data],
            'low': [c.low for c in self.data],
            'close': [c.close for c in self.data],
            'volume': [c.volume for c in self.data]
        }
        return pd.DataFrame(data_dict)

    def _prepare_data(self):
        """
        Prepares the data for training, scaling and reshaping it for LSTM input.
        """
        # Only use 'open', 'high', 'low', 'close', 'volume' for prediction
        data = self.df[['open', 'high', 'low', 'close', 'volume']].values
        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []

        for i in range(self.look_back, len(data_scaled)):
            X.append(data_scaled[i-self.look_back:i])  # Input sequence
            y.append(data_scaled[i, 3])  # Predict close price (index 3 in the columns)

        X = np.array(X)
        y = np.array(y)

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def build_model(self):
        """
        Builds the LSTM model for cryptocurrency price prediction using PyTorch.
        """
        self.model = LSTMModel(input_size=5, hidden_layer_size=50, num_layers=2, output_size=1)

    def train_model(self, epochs=10, batch_size=32, learning_rate=0.001):
        """
        Trains the LSTM model on the prepared data.
        
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size for training.
        :param learning_rate: Learning rate for optimizer.
        """
        criterion = nn.MSELoss()  # Mean Squared Error loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            output = self.model(self.X)
            loss = criterion(output, self.y.view(-1, 1))  # Reshape y to match output shape
            
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def predict(self, data):
        """
        Predicts the close price for the given data using the trained model.
        
        :param data: The data to predict on (list of ChartData).
        :return: The predicted closing price.
        """
        # Prepare data from the new input
        data = self._prepare_data_from_new_data(data)
        
        # Ensure the data only contains `look_back` data points
        if len(data) > self.look_back:
            data = data[-self.look_back:]  # Use the most recent `look_back` points
        
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        # Reshape the data to have the shape (1, look_back, 5)
        data_scaled = np.reshape(data_scaled, (1, self.look_back, 5))  # 1 sample, `look_back` sequence length, 5 features
        
        # Convert to tensor
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
        
        # Set the model to evaluation mode and predict
        self.model.eval()
        with torch.no_grad():
            predicted_price = self.model(data_tensor)
        
        # If predicted_price is a tensor with multiple values, get the first value (assuming 1 prediction output)
        predicted_price = predicted_price.detach().numpy().reshape(-1, 1)
        
        # Reshape the predicted price to have 5 features, as expected by the scaler
        predicted_price_reshaped = np.repeat(predicted_price, 5, axis=1)  # Repeat predicted value across 5 features
        
        # Inverse scale the prediction to return it to original scale
        predicted_price_original = self.scaler.inverse_transform(predicted_price_reshaped)
        
        # Return only the first column (the predicted price)
        return predicted_price_original[:, 0]
        



    def _prepare_data_from_new_data(self, new_data):
        """
        Prepares the new data for prediction.
        
        :param new_data: New list of ChartData objects to predict on.
        :return: Prepared and scaled data for prediction.
        """
        new_df = self._create_dataframe_from_new_data(new_data)
        data = new_df[['open', 'high', 'low', 'close', 'volume']].values
        return data

    def _create_dataframe_from_new_data(self, new_data):
        """
        Creates a DataFrame from the new ChartData for prediction.
        
        :param new_data: New list of ChartData objects.
        :return: DataFrame.
        """
        data_dict = {
            'timestamp': [c.timestamp for c in new_data],
            'open': [c.open for c in new_data],
            'high': [c.high for c in new_data],
            'low': [c.low for c in new_data],
            'close': [c.close for c in new_data],
            'volume': [c.volume for c in new_data]
        }
        return pd.DataFrame(data_dict)


# Define the LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=self.hidden_layer_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the last hidden state

        return out




