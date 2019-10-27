import pandas
import numpy as np


def load_data(data_path):
    data = pandas.read_csv(data_path)

    return data


def preprocess_data(data, top_n_items, opening_time):
    # select top n most frequent items
    items = data['Item Name'].value_counts()[:top_n_items]
    rows = data[data['Item Name'].isin(items.keys())]
    data = rows.reset_index(drop=True)

    # prepare features
    X = prepare_features(data, opening_time)

    # prepare labels
    y = prepare_labels(data, top_n_items)

    # train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                                        y,
    #                                                                        test_size=0.1,
    #                                                                        random_state=42)

    return data, X, y


def prepare_features(data, opening_time):
    # parse timestamp strings
    purchase_times = map(get_time, data['Order Date'])
    purchase_times = np.fromiter(map(get_min_of_day, purchase_times), int)
    X = purchase_times - opening_time
    X = np.expand_dims(X, axis=1)

    return X


def prepare_labels(data, top_n_items):
    # convert item names to categorical
    categorical = data['Item Name'].astype("category").cat.codes
    categorical = categorical.to_numpy()

    # one-hot encoding
    one_hot = np.zeros((categorical.shape[0], top_n_items), dtype=np.int16)
    one_hot[np.arange(categorical.shape[0]), categorical] = 1

    # replace one-hot with item quantity
    for index, row in data.iterrows():
        one_hot[index][categorical[index]] = row['Quantity']

    y = one_hot

    return y


def reduce_sparsity(top_n_items, X, y):
    non_sparse_X = list()
    non_sparse_y = list()

    for i in range(top_n_items):
        X_cls = X[y[:, i] != 0]
        y_cls = y[y[:, i] != 0, i]
        non_sparse_X.append(X_cls)
        non_sparse_y.append(y_cls)

    return non_sparse_X, non_sparse_y


def get_min_of_day(time):
    hour, min = time.split(":")
    return (int(hour) * 60) + int(min)


def get_time(timestamp):
    return timestamp.split(" ")[1]


def format_order_result(order):
    # conform to return format
    result = []
    for idx, x in enumerate(order):
        result.append([idx] * int(x))

    flatten = [item for sublist in result for item in sublist]

    return int(np.sum(order)), np.array(flatten)
