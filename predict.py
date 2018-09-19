import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def element_to_date_time(element):
    """
    Transforms a string into datetime object
    :param element: a date string of format year-month-day"
    :return: a datetime object
    """
    return datetime(int(element.split('-')[0]),
                    int(element.split('-')[1]),
                    int(element.split('-')[2]))


def to_date_time(data, column_name):
    """
    Transforms a dataframe's column of strings into datetime objects
    :param data: dataframe
    :param column_name: name of the column to be transformed
    :return: dataframe with the column transformed from string to datetime
    """
    data[column_name] = [element_to_date_time(x) for x in data[column_name]]
    return data


def columns_to_datetime(data):
    """
    Changes the 'snapshot_date' and 'flight_departure_date' columns from string to datetime.
    :param data: dataframe that has two columns 'snapshot_date' and 'flight_departure_date'.
    :return: dataframe with transformed columns
    """
    data = to_date_time(data, 'snapshot_date')
    data = to_date_time(data, 'flight_departure_date')
    data = data.sort_values(['flight_departure_date', 'snapshot_date'],
                            ascending=[True, True])
    return data


def get_tickets_per_day(tickets_by_day, span, training):
    """
    Detrending the tickets sold on each day to reflect only the tickets specifically on that day (not cumulative)
    :param tickets_by_day: original list that has the cumulative sum of tickets sold up until a snapshot date
    :param span: number of days included to remove the trend from
    :param training: if true it leaves the last column (response variable) untouched
    :return: a list of tickets sold on each day
    """
    if training:
        tickets_per_day = [tickets_by_day[i + 1] - tickets_by_day[i] for i in range((span - 2))]
        tickets_per_day = [tickets_by_day[0]] + tickets_per_day + [tickets_by_day[-1]]
    else:
        tickets_per_day = [tickets_by_day[i + 1] - tickets_by_day[i] for i in range((span - 1))]
        tickets_per_day = [tickets_by_day[0]] + tickets_per_day
    return tickets_per_day


def transform_data_into_supervised(data, departure_dates, span=80, training=True):
    """
    Formats the data in a way to be used by supervised learning algorithms in scikit-learn, where each row corresponds
    to the 79 past snapshot days before the departure date (predictors) and if training set true, then last column would
    be the target variable (y - final number of tickets sold for the train leaving on departure date)
    :param data: Origin data in pandas dataframe format with 3 columns (departure date, snapshot date, tickets sold)
    :param departure_dates: a list that includes all the departure times present in the data
    :param span: total number of dates where tickets sold have been logged
    :param training: if true it formats the data for training (last column being the target variable and labelled y)
    :return: returns the formatted dataframe where each row is an observation for a departure date (given the example
    training set of shape 365, 80)
    """
    column_vec = ["day " + str(x) for x in range(1, span)]
    if training:
        column_vec.append('y')
    else:
        column_vec.append('day ' + str(span))
    data_supervised = pd.DataFrame(columns=column_vec, dtype='int64')
    for departure_date in departure_dates:
        tickets_by_day = list(data[data.flight_departure_date == departure_date]['tickets_sold_by_snapshot_date'].values)
        data_supervised.loc[departure_date] = get_tickets_per_day(tickets_by_day, span, training)
    return data_supervised


def scale_data(X, training=True, span=79):
    """
    Using min max scaling on data so that every value would be between 0 and 1.
    :param X: Data to be scaled
    :param training: if True it trains 79 different scalers and stores them before transforming the data
    :param span: the length of the interval where snapshots has been made (1 to 79)
    :return: scaled dataset
    """
    if training:
        for i in range(span):
            columns = list(range(i+1)) + list(range(span, span+6))
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit(X[X.columns[columns]])
            fh = open(f"preprocessing/scaler_{len(columns)}_features", "wb")
            pickle.dump(min_max_scaler, fh)
    else:
        fh = open(f"preprocessing/scaler_{X.shape[1]}_features", "rb")
        min_max_scaler = pickle.load(fh)
    return min_max_scaler.transform(X)


def one_hot_encoder(data, fit_on):
    """
    One hot encoding of categorical data
    :param data: array to be encoded
    :param fit_on: list of values that shows the range of possible categorical values for the one hot encoding training
    :return: one hot encoded representative if data
    """
    enc = preprocessing.OneHotEncoder()
    enc.fit(fit_on)
    return enc.transform(data).toarray()


def add_temporal_features(X, dates, training=True):
    """
    Add temporal metadata as added features to provide more accurate models
    :param X: training/test data where metadata should get added to
    :param dates: list of departure dates that temporal meta data will get extracted from
    :param training: if true scalers will get trained before min max scaling the data
    :return: training/test data with temporal metadata added to it
    """
    X = X.assign(quarter_start=dates.is_quarter_start)
    X = X.assign(quarter_end=dates.is_quarter_end)
    X = X.assign(month_start=dates.is_month_start)
    X = X.assign(month_end=dates.is_month_end)
    X = X.assign(dayofmonth=dates.day)
    X = X.assign(weekofyear=dates.weekofyear)
    X_temp = scale_data(X, training)
    dayofweek = one_hot_encoder(np.array(dates.dayofweek).reshape(-1, 1), fit_on=np.arange(0, 7).reshape(-1, 1))
    month = one_hot_encoder(np.array(dates.month).reshape(-1, 1), np.arange(1, 13).reshape(-1, 1))
    X_temp = np.concatenate((X_temp, dayofweek, month), axis=1)
    X = pd.DataFrame(X_temp, index=X.index)
    return X


def train_model_up_to_date(X, y, current_date):
    """
    Training models and make them persistent
    :param X: Training data (predictors)
    :param y: Target data
    :param current_date: a model will get trained only using the data available up to current_date sales
    :return: None - stores the models on the disk.
    """
    svr = svm.SVR(C=15, kernel='linear')
    scores = cross_val_score(svr, X, y, cv=10,
                             scoring='neg_mean_squared_error')
    LOGGER.info("Trained the model for cases with %s day of sales information, with 10-fold cross validation score of %s (mean squared error)",
                current_date,
                np.abs(np.mean(scores)))
    svr.fit(X, y)
    days_until_departure = 80 - current_date
    LOGGER.info("Storing the fitted model for flights departing in %s days.",
                days_until_departure)
    fh = open(f"models/flights_departing_in_{days_until_departure}_model", "wb")
    pickle.dump(svr, fh)


def train(training_data_path="sales_data.csv"):
    """
    Trains the data on given data available from the path. It trains multiple models for data available up to each
    snapshot date. In this case 79 different models.
    :param training_data_path: path to the training data
    :return: None - models will get stored on the drive
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("preprocessing"):
        os.makedirs("preprocessing")
    data = pd.read_csv(training_data_path)
    data = columns_to_datetime(data)
    departure_dates = sorted(list(set(data.flight_departure_date)))
    data_supervised = transform_data_into_supervised(data, departure_dates)
    dates = pd.DatetimeIndex(data_supervised.index.values)
    X = data_supervised[data_supervised.columns.values[:-1]]
    X = add_temporal_features(X, dates)
    y = data_supervised.y
    for i in range(79):
        columns = list(range(i+1)) + list(range(79, 104))
        train_model_up_to_date(X[X.columns[columns]], y, (i+1))


def predict(departure_date, current_date, path):
    """
    predicts the number tickets to be sold for the train departing on departure date given the data available up to
    current_data
    :param departure_date: number tickets sold for this departure date will get predicted
    :param current_date: snapshots are available up to this date
    :param path: path to the data that predictions should get made.
    :return: returns the rounded estimate to the closest integer.
    """
    if not os.path.exists(path):
        LOGGER.error("Path to date is invalid. Path doesn't exist.")
        exit()
    data = pd.read_csv(path)
    data = data[data['flight_departure_date'] == departure_date]
    data = columns_to_datetime(data)
    current_date_datetime = element_to_date_time(current_date)
    departure_date_datetime = element_to_date_time(departure_date)
    data = data[data['snapshot_date'] <= current_date_datetime]
    X = transform_data_into_supervised(data, [departure_date_datetime], data.shape[0], False)
    dates = pd.DatetimeIndex(X.index.values)
    X = add_temporal_features(X, dates, False)
    fh = open(f"models/flights_departing_in_{80-data.shape[0]}_model", "rb")
    model = pickle.load(fh)
    return np.round(model.predict(X)[0])


if __name__ == "__main__":
    # If path models exists, then the models are previously trained and there's no need for the training step.
    # To retrain the models, models directory should be deleted.
    # If new training data is going to be used it should be put in the same directory with name "sales_data.csv".
    LOGGER.info("Training the models")
    if not os.path.exists("./models"):
        train()

    if len(sys.argv) != 4:
        LOGGER.info(
            """arg(s) for prediction are missing. Will not continue into predicting.
            The following arguments should be passed to the script with the same order:
            Plane Departure date with the following format: "year-month-day"
            Current date with the following format: "year-month-day"
            Path to data.

            Usage: python predict.py 2017-02-01 2017-01-10 /path_to_file/sales_data.csv"""
        )
    else:
        departure_date = sys.argv[1]
        current_date = sys.argv[2]
        path = sys.argv[3]

        # Predicting sales
        predicted_sales = predict(departure_date, current_date, path)
        LOGGER.info("Estimated tickets to be sold for departure date %s -> %s", departure_date, predicted_sales)
