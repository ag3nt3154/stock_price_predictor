import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import yfinance as yf

def generate_stock_vector(ticker, stock_data_df, stock_intrinsic_factors):
    """
    Generate input vector from stock info dataframe
    """
    stock = stock_data_df.loc[stock_data_df['symbol'] == ticker]
    stock_vector = [float(stock[f].values[0]) for f in stock_intrinsic_factors]
    stock_vector = np.array(stock_vector)
    return stock_vector


def rec_key_change(key):
    """
    Change recommendation key to numerical values
    """
    if key == 'none':
        return 0
    elif key == 'hold':
        return 1
    elif key == 'buy':
        return 2
    elif key == 'strong_buy':
        return 3

def stock_sector_change(key, stock_sectors):
    """
    Change stock sector into numerical values
    """
    return stock_sectors.index(key)


class stock_price_predictor:
    def __init__(self, intrinsic_factors_csv='stock_intrinsic_factors.json'):
        # Read important intrinsic factors
        # Factors that affect the intrinsic value of a stock
        # List of factors to be set for the vector input
        with open(intrinsic_factors_csv, 'r') as file:
            self.stock_intrinsic_factors = json.load(file)


        with open(intrinsic_factors_csv, 'r') as file:
            self.stock_sectors = json.load(file)
        self.stock_sectors.sort()
        

    def initialize_dataset(self, training_data_csv_file='stock_data.csv'):
        '''
        Read and process dataset from csv file
        '''
        # Read training data from csv
        self.stock_data_df = pd.read_csv(training_data_csv_file)

        # Generate list of sectors represented in training data 
        self.stock_sectors = self.stock_data_df['sector'].values
        self.stock_sectors = list(np.unique(self.stock_sectors))
        self.stock_sectors.sort()
        with open('stock_sectors.json', 'w') as file:
            json.dump(list(self.stock_sectors), file)

        # Change sectors into numerical values
        self.stock_data_df['sector'] = self.stock_data_df.apply(lambda row: stock_sector_change(row['sector'], self.stock_sectors), axis=1)
        # Change recommendationKey into numerical values
        self.stock_data_df['recommendationKey'] = self.stock_data_df.apply(lambda row: rec_key_change(row['recommendationKey']), axis=1)


    def generate_mean_variance(self, x_data):
        '''
        Normalise inputs to have mean 0 and variance 1.
        '''
        # Normalise inputs
        x_mean = []
        x_variance = []
        for i in range(len(x_data[0])):
            x_mean.append(np.mean([f[i] for f in x_data]))
            x_variance.append(np.var([f[i] for f in x_data]))
        x_mean = np.array(x_mean)
        x_variance = np.array(x_variance)

        self.x_mean = x_mean
        self.x_variance = x_variance

        # Save mean and variance into json file
        with open('mean_variance.json', 'w') as file:
            json.dump({'mean': list(x_mean), 'variance': list(x_variance)}, file)

        return x_data


    def generate_training_set(self):
        '''
        Generate the input vector from the stock data collected
        Normalise inputs to have mean 0 and variance 1
        Split dataset into training and test sets
        '''
        # Convert to np array
        x_data = np.array([generate_stock_vector(f, self.stock_data_df, self.stock_intrinsic_factors) \
            for f in self.stock_data_df['symbol'].values])
        
        prices = self.stock_data_df['currentPrice'].values
        # recKey = self.stock_data_df['recommendationKey'].values
        y_data = np.array([prices[i] for i in range(len(prices))])
        self.data_set = [x_data, y_data]

        self.generate_mean_variance(x_data)
        x_data = list(x_data)
        x_data = [(f - self.x_mean) / (np.sqrt(self.x_variance)) for f in x_data]
        x_data = np.array(x_data)

        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)

        
        self.training_set = [x_train, y_train]
        self.test_set = [x_test, y_test]

    def create_model(self):
        '''
        Create tensorflow model
        Create path linked to previously saved model weights
        '''
        # Set model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='relu')
            ])

        # Prepare to save the model weights
        self.checkpoint_path = "training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)


    def training(self):
        '''
        Train tensorflow model
        '''
        # Check TensorFlow version
        print("TensorFlow version:", tf.__version__)

        # Create ML model
        self.create_model()
        
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        # Define loss function
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Compile model
        self.model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['mse'])
        
        [x_train, y_train] = self.training_set
        [x_test, y_test] = self.test_set

        # Train model
        self.model.fit(x_train, y_train, epochs=50, callbacks=[cp_callback])

        # Evaluate model on test set
        self.model.evaluate(x_test,  y_test, verbose=2)
        print('---------------------------')
        predictions = self.model(x_test[:5]).numpy()
        for i in range(len(predictions)):
            print(f'Model: {predictions[i]}, Actual: {y_test[i]}')
        print('---------------------------')

        print(self.model.summary())
    

    def load_weights(self):
        '''
        Load saved weights from previous training
        '''

        # Loads the weights
        self.model.load_weights(self.checkpoint_path)

        # Define loss function
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Compile model
        self.model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['mse'])

        # Re-evaluate the model
        loss, acc = self.model.evaluate(self.test_set[0],  self.test_set[1], verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


    def predict_price(self, ticker):
        '''
        Use the ML model to predict the price of a stock given its fundamental info such as P/E ratio, cashflow, etc.
        Fundamental info taken from yahoo finance with yfinance.
        '''
        stock = yf.Ticker(ticker)
        # print(stock.info)
        stock_info_df = pd.DataFrame([stock.info])
        stock_vector = [generate_stock_vector(ticker, stock_info_df, self.stock_intrinsic_factors)]
        stock_vector = np.array([(f - self.x_mean) / (np.sqrt(self.x_variance)) for f in stock_vector])
        price = self.model(stock_vector).numpy()[0][0]
        return price
        
                

if __name__ == '__main__':

    # # Training Demonstration
    # predictor = stock_price_predictor()
    # predictor.initialize_dataset()
    # predictor.generate_training_set()
    # predictor.training()

    # Stock price prediction demonstration
    predictor = stock_price_predictor()
    predictor.initialize_dataset()
    predictor.generate_training_set()
    predictor.create_model()
    predictor.load_weights()
    ticker = 'MSFT'
    price = predictor.predict_price(ticker=ticker)
    print(f'Actual Price: {yf.Ticker(ticker).info["currentPrice"]}')
    print(f'Predicted Price: {price}')



    



    
