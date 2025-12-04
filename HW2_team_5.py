import marimo

__generated_with = "unknown"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ML hyperparametors constructor
    """)
    return
    

@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest
    """)
    return
    

@app.cell(hide_code=True)
def _(mo):
    mo.image(src="gridSearchRandForest.jpg")
    return


@app.cell
def _(mo):
    d_rfr = mo.ui.dictionary({
                            "n_estimators": mo.ui.slider(5,1500, step= 5),
                            "max_depth": mo.ui.slider(5,50, step= 5),
                            "criterion": mo.ui.dropdown(options = ['squared_error', 'absolute_error', 'friedman_mse','poisson'],allow_select_none=False, value= 'squared_error')})
    d_rfr
    return (d_rfr,)


@app.cell(hide_code=True)
def _(
    RandomForestRegressor,
    X_test,
    X_train,
    d_rfr,
    cross_validate,
    cross_val_predict,
    np,
    plt,
    y_test,
    y_train,
):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_validate, cross_val_predict

    modelRFR2_marimo = RandomForestRegressor(random_state=13, n_estimators= d_rfr['n_estimators'].value, max_depth= d_rfr['max_depth'].value, criterion= d_rfr['criterion'].value)
    scores = cross_validate(modelRFR2_marimo, X_train, y_train, cv=5,
                            scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error'))
    print(f"MAE: {-scores['test_neg_mean_absolute_error'].mean()}")
    print(f"MSE: {-scores['test_neg_mean_squared_error'].mean()}")
    print(f"RMSE: {-scores['test_neg_root_mean_squared_error'].mean()}")
    print(f"MAPE: {-scores['test_neg_mean_absolute_percentage_error'].mean()}")
    print(f"R^2: {scores['test_r2'].mean()}")

    modelRFR2_marimo.fit(X_train, y_train)
    y_predRFR2_marimo = modelRFR2_marimo.predict(X_test)
    _nn = np.arange(min(y_test), max(y_test), 0.01)
    _fig, _ax = plt.subplots()
    _ax.scatter(y_test, y_predRFR2_marimo)
    _ax.plot(_nn, _nn, color='red')
    plt.xlabel('Real results')
    plt.ylabel('Prediction results')
    plt.title('Comparing the real and prediction results for RandomForestRegressor with choosing parameters')
    _ax.fill_between(_nn, _nn + 1, _nn - 1, color='green', alpha=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # User Dashboard
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    year = mo.ui.dropdown(options= [i for i in range(2015, 2025)],allow_select_none=False, value= 2015)
    mo.md(f"Please choose year: {year}")
    return (year,)


@app.cell(hide_code=True)
def _(mo, regions):
    Region = mo.ui.dropdown(options=regions.keys(),allow_select_none= False, value= "Central and Eastern Europe")
    mo.md(f"Please choose Region: {Region}")
    return (Region,)


@app.cell(hide_code=True)
def _(Region, dfCode, plt, year):
    pie = dfCode[(dfCode.Year == year.value) & (dfCode['Regional indicator'] == Region.value)]

    plt.bar(pie.Country.tolist(), pie['Happiness score'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Country happiness in region {Region.value} in {year.value}')
    plt.show()
    return


@app.cell(hide_code=True)
def _(Region, dfCode, mo, year):
    cntrylist = dfCode[(dfCode['Regional indicator'] == Region.value) & (dfCode.Year == year.value)].Country.unique()
    Country = mo.ui.dropdown(options= cntrylist, allow_select_none= False,value=cntrylist[0])

    mo.md(f"Please choose Country: {Country}")
    return (Country,)


@app.cell(hide_code=True)
def _(Country, df, dfCode, plt, year):
    print("Happiness level for", Country.value, "in", year.value, "is", *dfCode[(df['Country'] == Country.value) & (df.Year == year.value)]['Happiness score'])
    countrydata = dfCode[(dfCode['Country'] == Country.value)]
    colors_bar = ['blue'] * len(countrydata.Year) 
    colors_bar[year.value - 2015] = 'green'
    bar = plt.bar(countrydata.Year, countrydata['Happiness score'], color= colors_bar)
    plt.title(f"Happiness level for {Country.value}")
    plt.ylim(min(countrydata['Happiness score']) -0.1, max(countrydata['Happiness score'])+0.1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(Region, dfCode, mo):
    d = mo.ui.dictionary({
    "GDP per capita": mo.ui.slider(0, 10, step= 1, value= dfCode[dfCode[f"Region_{Region.value}"] == True]['GDP per capita'].mean()),
    "Social support": mo.ui.slider(0, 1, step= 0.01, value= dfCode[dfCode[f"Region_{Region.value}"] == True]['Social support'].mean()),
    "Healthy life expectancy": mo.ui.slider(35, 90, step= 1, value= dfCode[dfCode[f"Region_{Region.value}"] == True]['Healthy life expectancy'].mean()),
    "Freedom to make life choices": mo.ui.slider(0, 1, step= 0.01, value = dfCode[dfCode[f"Region_{Region.value}"] == True]['Freedom to make life choices'].mean()),
    "Generosity": mo.ui.slider(0, 1, step= 0.01, value= dfCode[dfCode[f"Region_{Region.value}"] == True]['Generosity'].mean()),
    "Perceptions of corruption": mo.ui.slider(0, 1, step= 0.01, value= dfCode[dfCode[f"Region_{Region.value}"] == True]['Perceptions of corruption'].mean() )})

    mo.md(f"Choose user input data for predicting happiness in {Region.value}: {d}")
    return (d,)


@app.cell(hide_code=True)
def _(Region, d, dfCode, regions):
    user_data ={"GDP per capita": d['GDP per capita'].value,
                "Social support": d['Social support'].value,
                "Healthy life expectancy": d['Healthy life expectancy'].value,
                "Freedom to make life choices":d['Freedom to make life choices'].value,
                "Generosity": d['Generosity'].value,
                "Perceptions of corruption": d['Perceptions of corruption'].value}
    for regs in regions.keys():
        s = f"Region_{regs}"
        if Region.value == regs:             
            user_data[s] = True
        else:
            user_data[s] = False
    print("Input data is", user_data)
    print("Average values for chosen region", Region.value)
    print(dfCode[dfCode[f"Region_{Region.value}"] == True][["GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices", "Generosity", "Perceptions of corruption"]].mean())
    return (user_data,)


@app.cell(hide_code=True)
def _(Region, X_train, dfCode, modelRFR2, pd, user_data):
    user_test = pd.DataFrame(user_data, index=[0])
    y_user = modelRFR2_marimo.predict(user_test[X_train.columns])
    print("Predicted Happiness level:", round(*y_user, 2))
    print('Current average happiness', round(dfCode[dfCode[f"Region_{Region.value}"] == True]['Happiness score'].mean(), 2) )
    return


if __name__ == "__main__":
    app.run()
