import datetime as dt
import sys
import matplotlib.pyplot as plt
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

def feature_selection(train_data, cv_data, target_data, cv_target_data):
    """Select the optimal number of features to use."""
    max_features = 0
    min_mse = sys.maxsize
    max_r2 = 0
    plt.ion()
    fig, ax1 = plt.subplots()
    df = pd.DataFrame({'features':[], 'r2':[], 'mse':[]})
    for i in range(1, train_data.shape[1]):
        clf = RandomForestRegressor(max_features=i, random_state=1000)

        clf.fit(train_data, target_data)
        cv_predictions = clf.predict(cv_data)
        cv_predictions = np.maximum(0, cv_predictions)
        mse = mean_squared_error(cv_target_data, cv_predictions)
        r2 = r2_score(cv_target_data, cv_predictions)
        
        df2 = pd.DataFrame({'features':[i], 'r2':[r2], 'mse':[mse]})
        df = df.append(df2)
        
        ax1.plot(df.features, df.mse, 'b-')
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Mean Squared Error', color='b')

        ax2 = ax1.twinx()
        ax2.plot(df.features, df.r2, 'r-')
        ax2.set_ylabel('R2 Score', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        plt.draw()
        ax1.cla()
        ax2.cla()

        if mse < min_mse and r2 > max_r2:
            min_mse = mse
            max_r2 = r2
            max_features = i

    return max_features

def main():
    train_df = munge_data('./data/train.csv')
    target_df = train_df['count']
    train_df = train_df.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
    test_df = munge_data('./data/test.csv')

    scaler = preprocessing.StandardScaler()
    pnf = preprocessing.PolynomialFeatures()
    
    train_data = scaler.fit_transform(train_df.values)
    train_data = pnf.fit_transform(train_data)
    target_data = target_df.values
    test_datetimes = test_df['datetime'].map(lambda ts: ts.strftime('%Y-%m-%d %H:%M:%S'))
    test_df = test_df.drop(['datetime'], axis=1)
    test_data = scaler.transform(test_df.values)
    test_data = pnf.transform(test_data)
        
    train_data, cv_data, target_data, cv_target_data = cross_validation.train_test_split(
        train_data, target_data, test_size=0.2, random_state=1000)

    max_features = 44

    # max_features = feature_selection(train_data, cv_data, target_data, cv_target_data)

    # print('Optimal number of features: {}'.format(max_features))

    clf = RandomForestRegressor(max_features=max_features, random_state=1000)

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