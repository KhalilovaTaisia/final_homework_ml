import datetime

import dill
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.impute import SimpleImputer


def filter_data(df):
    columns_to_drop = [
        'session_id',
        'client_id',
        'utm_keyword',
        'device_model'
    ]
    return df.drop(columns_to_drop, axis=1)


def creating_new_features(df):
    import pandas
    df_copy = df.copy()

    df_copy.visit_date = pandas.to_datetime(df_copy.visit_date, utc=True)
    df_copy.visit_time = pandas.to_datetime(df_copy.visit_time, utc=True)
    df_copy['visit_hour'] = df_copy.visit_time.apply(lambda x: x.hour)
    df_copy['visit_month'] = df_copy.visit_date.apply(lambda x: x.month)
    df_copy['visit_day'] = df_copy.visit_date.apply(lambda x: x.day)
    df_copy['visit_weekday'] = df_copy.visit_date.apply(lambda x: x.weekday())

    def season_of_visit(month):
        if (month <= 2) | (month > 11):
            season = 'winter'
        elif (month <= 5):
            season = 'spring'
        elif (month <= 8):
            season = 'summer'
        else:
            season = 'autumn'
        return season

    df_copy['visit_season'] = df_copy.visit_month.apply(lambda x: season_of_visit(x))

    moscow_region_cities = ['Aprelevka',
                            'Balashikha',
                            'Bronnitsy',
                            'Vereya',
                            'Prominent',
                            'Volokolamsk',
                            'Voskresensk',
                            'Vysokovsk',
                            'Golitsyno',
                            'Dedovsk',
                            'Dzerzhinsky',
                            'Dmitrov',
                            'Dolgoprudny',
                            'Domodedovo',
                            'Drezna',
                            'Dubna',
                            'Egorievsk',
                            'Zhukovsky',
                            'Zaraisk',
                            'Zvenigorod',
                            'Ivanteevka',
                            'Istra',
                            'Kashira',
                            'Wedge',
                            'Kolomna',
                            'Korolev',
                            'Kotelniki',
                            'Krasnoarmeysk',
                            'Krasnogorsk',
                            'Krasnozavodsk',
                            'Krasnoznamensk',
                            'Cuban',
                            'Kurovskoe',
                            'Likino-Dulyovo',
                            'Lobnya',
                            'Losino-Petrovsky',
                            'Lukhovitsy',
                            'Lytkarino',
                            'Lyubertsy',
                            'Mozhaysk',
                            'Mytishchi',
                            'Naro-Fominsk',
                            'Noginsk',
                            'Odintsovo',
                            'Lakes',
                            'Orekhovo-Zuevo',
                            'Pavlovsky Posad',
                            'Peresvet',
                            'Podolsk',
                            'Protvino',
                            'Pushkino',
                            'Pushchino',
                            'Ramenskoe',
                            'Reutov',
                            'Roshal',
                            'Ruza',
                            'Sergiev Posad',
                            'Serpukhov',
                            'Solnechnogorsk',
                            'Old Kupavna',
                            'Stupino',
                            'Taldom',
                            'Fryazino',
                            'Khimki',
                            'Khotkovo',
                            'Chernogolovka',
                            'Chekhov',
                            'Shatura',
                            'Schelkovo',
                            'Elektrogorsk',
                            'Elektrostal',
                            'Electrocoal',
                            'Yakhroma',
                            'Moscow'
                            ]
    df_copy['is_Moscow_region'] = df_copy.apply(lambda x: 1 if x.geo_city in moscow_region_cities else 0, axis=1)

    import pickle
    with open('data/coordinates_dict.pkl', 'rb') as file:
        coordinates = pickle.load(file)

    def lat(city):
        lat = coordinates[city][0]
        return lat

    def long(city):
        long = coordinates[city][1]
        return long

    df_copy['lat'] = df_copy.geo_city.apply(lambda x: lat(x))
    df_copy['long'] = df_copy.geo_city.apply(lambda x: long(x))
    return df_copy.drop(['visit_date', 'visit_time', 'geo_city', 'geo_country'], axis=1)


def outliers_visit_number(df):
    df_copy = df.copy()

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    boundaries = calculate_outliers(df_copy.visit_number)
    is_outlier = (df_copy.visit_number > boundaries[1])
    df_copy.loc[is_outlier, 'visit_number'] = int(boundaries[1])

    return df_copy


def main():
    print('Target Action Prediction Pipeline')

    df = pd.read_csv('data/df_full.csv', low_memory=False)

    X = df.drop(['target'], axis=1)
    y = df['target']

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('new_features', FunctionTransformer(creating_new_features)),
        ('outliers', FunctionTransformer(outliers_visit_number))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int32', 'int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    models = [
        LogisticRegression(verbose=True, random_state=42, max_iter = 200, C = 2),
        #RandomForestClassifier(bootstrap=False, n_jobs = -1, verbose= True, random_state = 42,
        #                       max_depth=100, max_features='log2', min_samples_leaf=3, min_samples_split=5),
        #MLPClassifier(verbose=True, random_state=42, activation='logistic', alpha=0.0001, hidden_layer_sizes=(100,20))
    ]

    best_score = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('preprocessor_transformer', preprocessor_transformer),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best_model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc_score: {best_score:.4f}')

    target_pipe = {
        'model': best_pipe,
        'metadata': {
            'name': 'Target action prediction model',
            'author': 'Khalilova Taisia',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'roc_auc_score': best_score
        }
    }

    with open('target_pipe.pkl', 'wb') as file:
        dill.dump(target_pipe, file)


if __name__ == '__main__':
    main()