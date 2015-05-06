import datetime as dt
import numpy as np
import pandas as pd
from sklearn import cross_validation, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def munge_data(csv_input):
    """Take train and test set and make them useable for machine learning algorithms."""
    df = pd.read_csv(csv_input)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date.map(dt.date.toordinal)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['dayoff'] = (((df['weekday'] == 5) & (df['weekday'] == 6)) | (df['holiday'] == 1)).astype(int)
    df.index = pd.DatetimeIndex(df['datetime'])
    return df


def main():
    train_df = munge_data('./data/train.csv')
    target_df = train_df['count']
    train_df = train_df.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
    test_df = munge_data('./data/test.csv')
    
    train_data = preprocessing.scale(train_df.values)
    target_data = target_df.values
    test_datetimes = test_df['datetime'].map(lambda ts: ts.strftime('%Y-%m-%d %H:%M:%S'))
    test_df = test_df.drop(['datetime'], axis=1)
    test_data = preprocessing.scale(test_df.values)

    clf = RandomForestRegressor(random_state=1000)
    
    train_data, cv_data, target_data, cv_target_data = cross_validation.train_test_split(
        train_data, target_data, test_size=0.2, random_state=1000)

    clf.fit(train_data, target_data)
    cv_predictions = clf.predict(cv_data)
    cv_predictions = np.maximum(0, cv_predictions)
    print(mean_squared_error(cv_target_data, cv_predictions))
    print(r2_score(cv_target_data, cv_predictions))
    predictions = clf.predict(test_data)
    predictions = np.maximum(0, predictions)


    with open('output.csv', 'w') as o:
        o.write('datetime,count\n')
        for test_datetime, prediction in zip(test_datetimes, predictions):
            o.write('{},{}\n'.format(test_datetime, prediction))

if __name__ == '__main__':
    main()