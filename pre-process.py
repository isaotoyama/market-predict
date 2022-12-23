import pandas as pd
data = pd.read_csv('impressions.csv')

import matplotlib.pyplot as plt
import seaborn as sns
quan = list(data.loc[:, data.dtypes != 'object'].columns.values)
grid = sns.FacetGrid(pd.melt(data, value_vars=quan),
                     col='variable', col_wrap=4, height=3, aspect=1,
                     sharex=False, sharey=False)
grid.map(plt.hist, 'value', color="steelblue")
plt.show()

sns.heatmap(data._get_numeric_data().astype(float).corr(),
            square=True, cmap='RdBu_r', linewidths=.5,
            annot=True, fmt='.2f').figure.tight_layout()
plt.show()
       

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_pipeline(data, output):
    """
    Preprocessing pipeline part 1: Transform full data frame
    Arguments: Pandas dataframe, output column (dependent variable)
    Returns: Modified dataframe
    """
    data = cost_per_metric(data, output) if 'cost_per' in output \
                                         else data[data[output] > 0]
    data = drop_columns(data, output, threshold=.5)
    data = data.dropna(axis='index')
    data = create_other_buckets(data, threshold=.1)
    data = one_hot_encode(data)
    return data

def split_pipeline(data, output):
    """
    Preprocessing pipeline part 2: Split data into variables
    Arguments: Pandas dataframe, output column (dependent variable)
    Returns: List of scaled and unscaled dependent and independent variables
    """
    y, X = data.iloc[:, 0], data.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([output], axis=1), data[output], test_size=.2, random_state=1)
    X_scaled, y_scaled, X_train_scaled, y_train_scaled, X_test_scaled, \
        y_scaler = scale(X, y, X_train, y_train, X_test)
    return [X, y, X_train, y_train, X_test, y_test, X_scaled, y_scaled,
            X_train_scaled, y_train_scaled, X_test_scaled, y_scaler]

def cost_per_metric(data, output):
    """Create 'cost_per_...' column and remove data where output is 0 or NaN"""
    metric = output.replace('cost_per_', '') + 's'
    data = data[data[metric] > 0]
    data.insert(0, output, [row['cost'] / row[metric]
                            for index, row in data.iterrows()])
    return data

def drop_columns(data, output, threshold=0.5):
    """Drop columns with more than threshold missing data"""
    rows = data[output].count()
    for column in list(data.columns):
        if data[column].count() < rows * threshold:
            data = data.drop([column], axis=1)
    return data

def create_other_buckets(data, threshold=0.1):
    """Put rare categorical values into other bucket"""
    categoricals = list(data.select_dtypes(include='object').columns)
    for column in categoricals:
        results = data[column].count()
        groups = data.groupby([column])[column].count()
        for bucket in groups.index:
            if groups.loc[bucket] < results * threshold:
                data.loc[data[column] == bucket, column] = 'other'
    return data

def one_hot_encode(data):
    """One-hot encode categorical data"""
    categoricals = list(data.select_dtypes(include='object').columns)
    for column in categoricals:
        if 'other' in data[column].unique():
            data = pd.get_dummies(data, columns=[column], prefix=[column],
                                  drop_first=False)
            data = data.drop([column + '_other'], axis=1)
        else:
            data = pd.get_dummies(data, columns=[column], prefix=[column],
                                  drop_first=True)
    return data

def scale(X, y, X_train, y_train, X_test):
    """Scale dependent and independent variables"""
    X_scaler, y_scaler = StandardScaler(), StandardScaler()

    X_scaled = X_scaler.fit_transform(X.values.astype(float))
    y_scaled = y_scaler.fit_transform(
        y.values.astype(float).reshape(-1, 1)).flatten()

    X_train_scaled = pd.DataFrame(data=X_scaler.transform(
        X_train.values.astype(float)), columns=X.columns)
    y_train_scaled = y_scaler.transform(
        y_train.values.astype(float).reshape(-1, 1)).flatten()

    X_test_scaled = pd.DataFrame(data=X_scaler.transform(
        X_test.values.astype(float)), columns=X.columns)

    return [X_scaled, y_scaled, X_train_scaled, y_train_scaled,
            X_test_scaled, y_scaler]


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)

model = sm.OLS(self.y_train, sm.add_constant(self.X_train)).fit()
print(model.summary())

linear_score = np.mean(cross_val_score(estimator=linear_regressor,
                       X=X_train, y=y_train, cv=5,
                       scoring=mean_relative_accuracy))

tree_parameters = [{'min_samples_leaf': list(range(2, 10, 1)),
                    'criterion': ['mae', 'mse'],
                    'random_state': [1]}]
tree_grid = GridSearchCV(estimator=DecisionTreeRegressor(),
                         param_grid=tree_parameters,
                         scoring=mean_relative_accuracy, cv=5,
                         n_jobs=-1, iid=False)
tree_grid_result = tree_grid.fit(X_train, y_train)
best_tree_parameters = tree_grid_result.best_params_
tree_score = tree_grid_result.best_score_

from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree.dot', 
                feature_names=X_train.columns)

forest_parameters = [{'n_estimators': helpers.powerlist(10, 2, 4),
                      'min_samples_leaf': list(range(2, 10, 1)),
                      'criterion': ['mae', 'mse'],
                      'random_state': [1], 'n_jobs': [-1]}]
forest_grid = GridSearchCV(estimator=RandomForestRegressor(),
                           param_grid=forest_parameters,
                           scoring=mean_relative_accuracy, cv=5,
                           n_jobs=-1, iid=False)
forest_grid_result = forest_grid.fit(X_train, y_train)
best_forest_parameters = forest_grid_result.best_params_
forest_score = forest_grid_result.best_score_

svr_parameters = [{'kernel': ['linear', 'rbf'],
                   'C': helpers.powerlist(0.1, 2, 10),
                   'epsilon': helpers.powerlist(0.01, 2, 10),
                   'gamma': ['scale']},
                  {'kernel': ['poly'],
                   'degree': list(range(2, 5, 1)),
                   'C': helpers.powerlist(0.1, 2, 10),
                   'epsilon': helpers.powerlist(0.01, 2, 10),
                   'gamma': ['scale']}]
svr_grid = GridSearchCV(estimator=SVR(),
                        param_grid=svr_parameters,
                        scoring=mean_relative_accuracy, cv=5,
                        n_jobs=-1, iid=False)
svr_grid_result = svr_grid.fit(X_train_scaled, y_train_scaled)
best_svr_parameters = svr_grid_result.best_params_
svr_score = svr_grid_result.best_score_

training_accuracies = {}
test_accuracies = {}
for regressor in regressors:
    if 'SVR' in str(regressor):
        regressor.fit(X_train_scaled, y_train_scaled)
        training_accuracies[regressor] = hel.mean_relative_accuracy(
            y_scaler.inverse_transform(regressor.predict(
                X_train_scaled)), y_train)
        test_accuracies[regressor] = hel.mean_relative_accuracy(
            y_scaler.inverse_transform(regressor.predict(
                X_test_scaled)), y_test)
    else:
        regressor.fit(X_train, y_train)
        training_accuracies[regressor] = hel.mean_relative_accuracy(
            regressor.predict(X_train), y_train)
        test_accuracies[regressor] = hel.mean_relative_accuracy(
            regressor.predict(X_test), y_test)

     