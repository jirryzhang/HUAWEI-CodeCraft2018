import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



class mLSTM:

    def __init__(self, dataset, time_step=4, train_begin=0, train_end=50):
        self.dataset = np.array(dataset)
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []
        self.time_step = time_step

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # dataset = scaler.fit_transform(np.array(dataset).reshape(1, -1))

        for i in range(train_begin, train_end-time_step):
            self.x_train.append(self.dataset[i:i+time_step])
            self.y_train.append(self.dataset[i+time_step])

        for i in range(train_end-time_step, len(dataset)-time_step):
            self.x_test.append(self.dataset[i:i+time_step])
            self.y_test.append(self.dataset[i+time_step])

        self.x_train = np.reshape(self.x_train, (-1, time_step, 1))
        for i in range(len(self.x_train)):
            self.x_train[i] = self.gaussian_weighted(self.x_train[i])

        self.x_test = np.reshape(self.x_test, (-1, time_step, 1))
        for i in range(len(self.x_test)):
            self.x_test[i] = self.gaussian_weighted(self.x_test[i])

        self.y_train = np.reshape(self.y_train, (-1, 1))
        self.y_test = np.reshape(self.y_test, (-1, 1))

    def gaussian_weighted(self, data):
        for i in range(self.time_step):
            p = 0.00044
            w = math.exp(- math.pow(data[i] - data[-1], 2) / 2 * p)
            data[i] = w * float(data[i])
        return data

    def lstm_model(self):
        model = Sequential()
        # model.add(LSTM(100, input_shape=(self.time_step, 1), return_sequences=True))
        model.add(LSTM(200, input_shape=(self.time_step, 1), return_sequences=True))
        model.add(LSTM(100, input_shape=(self.time_step, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        keras.backend.set_value(model.optimizer.lr, 0.5 * keras.backend.get_value(model.optimizer.lr))
        model.fit(self.x_train, self.y_train, epochs=300, batch_size=1, verbose=2)

        train_predict = model.predict(self.x_train)
        plt.figure(0)
        plt.plot(self.y_train)
        plt.plot(train_predict)
        print(self.y_train[-1])
        print(train_predict[-1])
        # print(sum(self.y_train))
        # print(sum(train_predict))

        test_predict = []
        for i in range(len(self.x_test)):
            test_predict.append(model.predict(self.x_train[-1].reshape(-1, self.time_step, 1))[0][0])
            self.dataset = np.append(self.dataset, test_predict[-1])
            self.x_train = np.append(self.x_train, self.gaussian_weighted(self.dataset[-self.time_step:])).reshape(-1, self.time_step, 1)
        plt.figure(1)
        plt.plot(self.y_test)
        plt.plot(test_predict)
        plt.show()
        print(self.y_test[-1])
        print(test_predict[-1])
        # print(sum(self.y_test))
        # print(sum(test_predict))
