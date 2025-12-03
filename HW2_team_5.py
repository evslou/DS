import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 1: Predicting World Happiness Level Using Machine Learning
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder, np, pd, plt, sns, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyzing and preprocessing data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correcting data types
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("world_happiness_combined.csv", delimiter=";")
    return (df,)


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    df.replace(',', '.', regex=True, inplace=True)
    df_1 = df.astype({'Happiness score': float, 'GDP per capita': float, 'Social support': float, 'Healthy life expectancy': int, 'Freedom to make life choices': float, 'Generosity': float, 'Perceptions of corruption': float})
    return (df_1,)


@app.cell
def _(df_1):
    df_1['Regional indicator'].unique()
    return


@app.cell
def _(df_1):
    df_1.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Business task 1 visualisations
    """)
    return


@app.cell
def _(df_1):
    regions = {}
    for reg in ['Western Europe', 'North America and ANZ', 'Middle East and North Africa', 'Latin America and Caribbean', 'Southeast Asia', 'Central and Eastern Europe', 'East Asia', 'Commonwealth of Independent States', 'Sub-Saharan Africa', 'South Asia']:
        regions[reg] = df_1[df_1['Regional indicator'] == reg]['Happiness score'].to_numpy()
    return (regions,)


@app.cell
def _(plt):
    colors = [plt.cm.Pastel1(i) for i in range(10)]
    return (colors,)


@app.cell
def _(colors, plt, regions):
    plt.figure(figsize=(13, 7))
    bp = plt.boxplot(regions.values(), tick_labels=regions.keys(), patch_artist=True)

    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)
    plt.title('Happiness score by region')
    plt.xlabel('Regions')
    plt.ylabel('Happiness score')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    return


@app.cell
def _(df_1, plt):
    plt.boxplot(df_1['Happiness score'])
    plt.title('Happiness level boxplot')
    return


@app.cell
def _(df_1, plt):
    plt.figure(figsize=(13, 7))
    dfreg = df_1.groupby(['Regional indicator'])['Happiness score'].mean()
    plt.bar(dfreg.index, dfreg)
    plt.title('Happiness level by region')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    return


@app.cell
def _(df_1, plt):
    plt.hist(df_1['Happiness score'])
    plt.title('Happiness level histogram')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Business task 2 visualisation
    """)
    return


@app.cell
def _(df_1, plt):
    plt.boxplot(df_1['Healthy life expectancy'])
    plt.title('Life expectancy level boxplot')
    return


@app.cell
def _(df_1, plt):
    plt.hist(df_1['Healthy life expectancy'])
    plt.title('Life expectancy level histogram')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filling in the null values
    """)
    return


@app.cell
def _(df_1):
    df_1['Regional indicator'].value_counts()
    return


@app.cell
def _(df_1):
    df_1.loc[(df_1['Country'] == 'Greece') & (df_1['Year'] == 2017), 'Regional indicator'] = 'Western Europe'
    df_1.loc[(df_1['Country'] == 'Cyprus') & (df_1['Year'] == 2019), 'Regional indicator'] = 'Western Europe'
    df_1.loc[(df_1['Country'] == 'Gambia') & (df_1['Year'] == 2019), 'Regional indicator'] = 'Sub-Saharan Africa'
    return


@app.cell
def _(df_1):
    df_1.isna().any().any()
    return


@app.cell
def _(df_1):
    df_1[(df_1['Country'] == 'Greece') | (df_1['Country'] == 'Cyprus') | (df_1['Country'] == 'Gambia')]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multivariate analysis
    """)
    return


@app.cell
def _(df_1, sns):
    sns.heatmap(df_1.corr(numeric_only=True), annot=True, cmap='coolwarm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coding categorical variables
    1. One-Hot Encoding
    """)
    return


@app.cell
def _(df_1):
    df_1
    return


@app.cell
def _(df_1, pd):
    dfCode = df_1.join(pd.get_dummies(df_1['Regional indicator'], prefix='Region'))
    return (dfCode,)


@app.cell
def _(dfCode):
    dfCode.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Label Encoding:
    """)
    return


@app.cell
def _(LabelEncoder, dfCode):
    le = LabelEncoder()
    dfCode["Country_encoded"] = le.fit_transform(dfCode["Country"])
    dfCode.head(10)
    return


@app.cell
def _(dfCode, plt, sns):
    plt.figure(figsize=(20, 15))
    plt.title('Correlation map')
    sns.heatmap(dfCode.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementing Machine Learning methods:
    1. LinearRegression:
    """)
    return


@app.cell
def _(dfCode, train_test_split):
    X = dfCode[
            [
                'GDP per capita',
                'Social support',
                'Healthy life expectancy',
                'Freedom to make life choices',
                'Generosity',
                'Perceptions of corruption',
            ] + [col for col in dfCode.columns if 'Region_' in col]]
    y = dfCode['Happiness score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train):
    X_train.sample(5)
    return


@app.cell
def _(X_test, X_train):
    X_train.shape, X_test.shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    Default model training
    """)
    return


@app.cell
def _(X_test, X_train, np, y_test, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)}")
    print(f"R^2: {r2_score(y_test, y_pred)}")
    return (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
        y_pred,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Lasso regression with best alpha hyper parameter
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    from sklearn.linear_model import Lasso
    mapemin = 1
    minaplha = 0



    alpha = [0.0001, 0.0005 , 0.001, 0.01, 0.005 ]
    for a in alpha:
        model2 = Lasso(alpha=a)
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred2)
        if mape < mapemin:
            mapemin = mape
            minaplha = a

    model2 = Lasso(alpha=minaplha)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    print("alpha:", minaplha)
    print(f"MAE: {mean_absolute_error(y_test, y_pred2)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred2)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred2))}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred2)}")
    print(f"R^2: {r2_score(y_test, y_pred2)}")
    print()
    return Lasso, y_pred2


@app.cell
def _(np, plt, y_pred, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_pred)

    _ax.plot(_nn, _nn, color='red')

    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the real and prediction results for LinearRegression')
    return


@app.cell
def _(np, plt, y_pred2, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_pred2)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the real and prediction results for Lasso Regression at aplha 0.001')
    return


@app.cell
def _(np, plt, y_pred, y_pred2, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_pred, label= 'LinearRegression', edgecolor='darkblue', s=55)
    _ax.scatter(y_test, y_pred2, color='orangered', label= 'Lasso', alpha=0.7, edgecolor='darkred', s=25)
    _ax.plot(_nn, _nn, color='red')
    plt.legend()
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the defaul LinearRegression and Lasso')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. RandomForest:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Default random forest
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    from sklearn.ensemble import RandomForestRegressor

    modelRFR = RandomForestRegressor(random_state=13)
    modelRFR.fit(X_train, y_train)
    y_predRFR = modelRFR.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_predRFR)}")
    print(f"MSE: {mean_squared_error(y_test, y_predRFR)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_predRFR))}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_predRFR)}")
    print(f"R^2: {r2_score(y_test, y_predRFR)}")
    return RandomForestRegressor, modelRFR, y_predRFR


@app.cell
def _(clf):
    clf.best_params_
    return


@app.cell
def _(
    RandomForestRegressor,
    X_test,
    X_train,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    modelRFR2 = RandomForestRegressor(
        random_state=13, 
        n_estimators=1000, max_depth=10, 
        criterion='friedman_mse')

    modelRFR2.fit(X_train, y_train)
    y_predRFR2 = modelRFR2.predict(X_test)
    print('Best number of trees')
    print(f'MAE: {mean_absolute_error(y_test, y_predRFR2)}')
    print(f'MSE: {mean_squared_error(y_test, y_predRFR2)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_predRFR2))}')
    print(f'MAPE: {mean_absolute_percentage_error(y_test, y_predRFR2)}')
    print(f'R^2: {r2_score(y_test, y_predRFR2)}')
    return modelRFR2, y_predRFR2


@app.cell
def _(np, plt, y_predRFR, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_predRFR)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the real and prediction results for RandomForestRegressor')
    return


@app.cell
def _(np, plt, y_predRFR2, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_predRFR2)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title("Comparing the real and prediction results for RandomForestRegressor: \nn_estimators=1000, max_depth=10, criterion='friedman_mse")
    return


@app.cell
def _(np, plt, y_predRFR, y_predRFR2, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_predRFR, label='Default', color='dodgerblue', edgecolors="darkblue", s=50)
    _ax.scatter(y_test, y_predRFR2, color='orangered', label='Tuned', s=30, edgecolors="darkred", alpha=0.8)
    _ax.plot(_nn, _nn, color='red')
    plt.legend()
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the real and prediction results for RandomForestRegressor')
    return


@app.cell
def _(np, plt, y_predRFR, y_test):
    counts, bins = np.histogram(y_predRFR - y_test)
    _fig, _ax = plt.subplots()
    plt.title('Distribution of errors')
    plt.xlabel('(y_pred - y_test)')
    plt.ylabel('Count')
    _ax.hist(bins[:-1], bins, weights=counts)
    plt.show()
    return


@app.cell
def _(X_train, modelRFR):
    f_i = {}
    for feature in range(len(X_train.columns)):
        f_i[X_train.columns[feature]] = round(modelRFR.feature_importances_[feature], 6)
    for k, v in sorted(f_i.items(), key=lambda x: -x[1]):
        print(f'Feature - {k}, Importance: {v}')
    return


@app.cell
def _(X_train, modelRFR2):
    f_i_2 = {}
    for feature2 in range(len(X_train.columns)):
        f_i_2[X_train.columns[feature2]] = round(modelRFR2.feature_importances_[feature2], 6)
    for key, val in sorted(f_i_2.items(), key=lambda x: -x[1]):
        print(f'Feature - {key}, Importance: {val}')
    return


@app.cell
def _(modelRFR, plt):
    from sklearn import tree
    plt.figure(figsize=(40, 30))
    tree.plot_tree(modelRFR.estimators_[0], max_depth=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Gradient Boosting
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    from sklearn.ensemble import GradientBoostingRegressor

    modelGBR = GradientBoostingRegressor(random_state=13)
    modelGBR.fit(X_train, y_train)
    y_predGBR = modelGBR.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_predGBR)}")
    print(f"MSE: {mean_squared_error(y_test, y_predGBR)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_predGBR))}")
    print(f"R^2: {r2_score(y_test, y_predGBR)}")
    return GradientBoostingRegressor, y_predGBR


@app.cell
def _(
    GradientBoostingRegressor,
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    y_test,
    y_train,
):
    _modelGBR2 = GradientBoostingRegressor(random_state=13, criterion='squared_error', loss='huber', n_estimators=200, learning_rate=0.1)
    _modelGBR2.fit(X_train, y_train)
    y_predGBR2 = _modelGBR2.predict(X_test)
    print(f'MAE: {mean_absolute_error(y_test, y_predGBR2)}')
    print(f'MSE: {mean_squared_error(y_test, y_predGBR2)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_predGBR2))}')
    print(f'R^2: {r2_score(y_test, y_predGBR2)}')
    return (y_predGBR2,)


@app.cell
def _(np, plt, y_predGBR, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_predGBR)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title("Comparing the real and prediction results for GradientBoostingRegressor")
    return


@app.cell
def _(np, plt, y_predGBR2, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_predGBR2)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title("Comparing the real and prediction results for GradientBoostingRegressor: \ncriterion='squared_error', loss='huber', n_estimators=200, learning_rate=0.1")
    return


@app.cell
def _(np, plt, y_predGBR, y_predGBR2, y_test):
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_predGBR, label='Default', s=50, color='dodgerblue', edgecolors="darkblue")
    _ax.scatter(y_test, y_predGBR2, label='Tuned', s=30, edgecolors="darkred", alpha=0.8, color='orangered')
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title("Comparing the real and prediction results for GradientBoostingRegressor: \ncriterion='squared_error', loss='huber', n_estimators=200, learning_rate=0.1")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    alphas = mo.ui.slider(0.0001, 1, step= 0.0001)
    mo.md(f"Choose alpha value: {alphas}")
    return (alphas,)


@app.cell
def _(
    Lasso,
    X_test,
    X_train,
    alphas,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    np,
    plt,
    r2_score,
    y_test,
    y_train,
):
    model2_m = Lasso(alpha=alphas.value)
    model2_m.fit(X_train, y_train)
    y_pred2_m = model2_m.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_pred2_m)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred2_m)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred2_m))}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred2_m)}")
    print(f"R^2: {r2_score(y_test, y_pred2_m)}")
    print()
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    _ax.scatter(y_test, y_pred2_m)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the real and prediction results for LassoRegression at aplha')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### User Dashboard
    """)
    return


@app.cell
def _(mo):
    gdp = mo.ui.slider(0, 10, step= 1)
    socSup = mo.ui.slider(0, 1, step= 0.01)
    health = mo.ui.slider(35, 90, step= 1)
    freedom = mo.ui.slider(0, 1, step= 0.01)
    generosity = mo.ui.slider(0, 1, step= 0.01)
    corruption = mo.ui.slider(0, 1, step= 0.01)
    region = mo.ui.text()
    return


@app.cell
def _(mo):
    d = mo.ui.dictionary({"gdp": mo.ui.slider(0, 10, step= 1),
                         "socSup": mo.ui.slider(0, 1, step= 0.01),
                         "health": mo.ui.slider(35, 90, step= 1),
                         "freedom": mo.ui.slider(0, 1, step= 0.01),
                         "generosity": mo.ui.slider(0, 1, step= 0.01),
                         "corruption": mo.ui.slider(0, 1, step= 0.01),
                         "region": mo.ui.text()})
    d
    return (d,)


@app.cell
def _(d, regions):
    user_data ={"GDP per capita": d['gdp'].value,
                "Social support": d['socSup'].value,
                "Healthy life expectancy": d['health'].value,
                "Freedom to make life choices":d['freedom'].value,
               "Generosity": d['generosity'].value,
               "Perceptions of corruption": d['corruption'].value}
    if d['region'].value not in regions.keys():
        print('No such region. Try again. List of regions:', regions.keys())
    else:
        for regs in regions.keys():
            s= f"Region_{regs}"
            if d['region'].value == regs: 
                user_data[s] = True
            else:
                user_data[s] = False
    print(user_data)
    return (user_data,)


@app.cell
def _(X_train, modelRFR2, pd, user_data):
    user_test = pd.DataFrame(user_data, index=[0])
    model_features = X_train.columns

    y_user = modelRFR2.predict(user_test[model_features])
    print(y_user)
    return


if __name__ == "__main__":
    app.run()
