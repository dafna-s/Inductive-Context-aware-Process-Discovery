import config

import os
import pandas as pd


def get_power_set(s):
    power_set = [set()]
    for element in s:
        new_sets = []
        for subset in power_set:
            new_sets.append(subset | {element})
        power_set.extend(new_sets)
    power_set.remove(set())
    return power_set


def get_data():
    base_directory = os.path.abspath(os.curdir)
    config.base_directory = base_directory
    data_dir = os.path.join(config.base_directory, config.data_dir)
    df = pd.read_csv(os.path.join(data_dir, config.data_file), encoding="ISO-8859-1")
    return df


def time_to_min(df):
    for i, v in df.items():
        # print(df, type(df), i, v)
        time_list = v.split(':')
        df.loc[i] = float(time_list[0]) * 60 + float(time_list[1]) + float(time_list[2]) / 60.0
    return df


def time_to_hour(df):
    # for i, v in df.items():
    #     time_list = v.split(':')
    #     # df.loc[i] = float(time_list[0]) + float(time_list[1]) / 60.0 + float(time_list[2]) / 360.0
    #     df.loc[i] = float(time_list[0]) + float(time_list[1]) / 60.0
    time = pd.DatetimeIndex(df)
    time = time.hour + time.minute / 60
    return pd.Series(time)


def min_to_time(minutes):
    return '{:02d}:{:02d}'.format(*divmod(int(float(minutes)), 60))


def hour_to_time(hour):
    ihours = round(hour-0.5)
    return "%02d:%02d" % (ihours, (hour - ihours) * 60)


def pre_process(df):
    df['time'] = time_to_hour(df['time'])
    # df['time'] = (df['time'] - min(df['time'])) / (max(df['time']) - min(df['time']))
    # df['duration'] = (df['duration'] - min(df['duration'])) / (max(df['duration']) - min(df['duration']))
    # df = df.drop(['duration'], axis=1)
    df_for_cluster = df.copy()
    # df_for_cluster['activity_orig'] = df_for_cluster['activity']
    # df_for_cluster = pd.get_dummies(df_for_cluster, columns=['resource', 'activity_orig'], prefix=['resource', 'activity'])

    # df_for_cluster = pd.get_dummies(df_for_cluster, columns=['resource', 'case id'])
    # if 'duration' in df.columns.values:
    #     df_for_cluster['duration'] = df_for_cluster['duration']/60.0
    if 'case id' in config.categorical:
        df_for_cluster['case id'] = df_for_cluster['case']
        df['case id'] = df['case']

    for feature in config.categorical:
        df_for_cluster[feature] = pd.factorize(df_for_cluster[feature])[0]
        temp_map = {}
        for orig_val in df[feature].unique():
            index = df.index[df[feature] == orig_val][0]
            new_val = df_for_cluster.iloc[index][feature]
            temp_map[new_val] = orig_val
        config.categorical_map.append(temp_map)


    # df_for_cluster = df_for_cluster.drop(['case'], axis=1)
    print(df_for_cluster)

    return df_for_cluster