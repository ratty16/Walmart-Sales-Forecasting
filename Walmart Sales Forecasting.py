#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy matplotlib seaborn plotly scikit-learn jovian opendatasets graphviz --upgrade --quiet')


# In[2]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize'] = (10,8)


# In[11]:


stores = pd.read_csv("/Users/ratty/Downloads/walmart-recruiting-store-sales-forecasting/stores.csv")
features = pd.read_csv("/Users/ratty/Downloads/walmart-recruiting-store-sales-forecasting/features.csv")
train = pd.read_csv("/Users/ratty/Downloads/walmart-recruiting-store-sales-forecasting/train.csv")
test = pd.read_csv("/Users/ratty/Downloads/walmart-recruiting-store-sales-forecasting/test.csv")


# In[12]:


display(stores.sample(5).sort_values(by='Store').reset_index(drop=True))


# In[13]:


display(features.sample(5).sort_values(by='Store').reset_index(drop=True))


# In[14]:


display(train.sample(5).sort_values(by='Store').reset_index(drop=True))


# In[16]:


display(test.sample(5).sort_values(by='Store').reset_index(drop=True))


# In[ ]:


features = features.merge(stores, how = 'left', on = 'Store')
def return_merged_data(data):
    df = data.merge(features, how = 'left' on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date'].reset_index(drop = True))


# In[ ]:


train_data, test_data = return_merged_data(train), return_merged_data(test)


# In[ ]:


display(train_data.sample(5).sort_values(by = 'Store').reset_index(drop = True))


# In[ ]:


def convert_boolean(data):
    data['IsHoliday'] = data['IsHoliday'].map({False : 0, True : 1}).astype('int')
    return data


# In[ ]:


train_data, test_data = convert_boolean(train_data), convert_boolean(test_data)


# In[ ]:


train_data.isna().sum()


# In[ ]:


def replace_nan(data):
    
    columns_to_fill = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    data[columns_to_fill] = data[columns_to_fill].fillna(0)
    
    return data


# In[ ]:


train_data, test_data = replace_nan(train_data), replace_nan(test_data)


# In[ ]:


def extract_date_info(df):
    df['Date2'] = pd.to_datetime(df['Date'])

    df['Day'] = df['Date2'].dt.day.astype('int')
    df['Month'] = df['Date2'].dt.month.astype('int')
    df['Year'] = df['Date2'].dt.year.astype('int')
#     df['DayOfWeek'] = df['Date2'].dt.weekday
    df['WeekOfYear'] = df['Date2'].dt.isocalendar().week.astype('int')
    df['Quarter'] = df['Date2'].dt.quarter.astype('int')
    df = df.drop(columns = ['Date2'])


    return df


# In[ ]:


train_data, test_data = extract_date_info(train_data), extract_date_info(test_data)


# In[ ]:


train_data.describe()


# In[ ]:


def markdown_info(df):
    
    # Drop rows with negative values in the specified columns
#     negative_mask = (df['MarkDown1'] < 0) | (df['MarkDown2'] < 0) | (df['MarkDown3'] < 0) | (df['MarkDown4'] < 0) | (df['MarkDown5'] < 0)
#     df = df[~negative_mask].copy()

    # Create a new column 'MarkDown' with the sum of values from the 5 columns
    df['MarkDown'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1, inplace=True)
    
    return df


# In[ ]:


train_data, test_data = markdown_info(train_data), markdown_info(test_data)


# In[ ]:


def isholiday(df):
    
    holiday_weeks = [1, 6, 36, 47, 52]
    df.loc[df['WeekOfYear'].isin(holiday_weeks), 'IsHoliday'] = 1
    
    return df


# In[ ]:


train_data = train_data[train_data['Weekly_Sales'] > 0]


# In[ ]:


train_data = train_data.sort_values(by=['Date']).reset_index(drop=True)


# In[ ]:


numeric_cols = ['Store', 'Dept', 'IsHoliday', 'Temperature','Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Day', 'Month','Year', 'WeekOfYear', 'Quarter', 'MarkDown','Weekly_Sales']


# In[ ]:


sns.heatmap(train_data[numeric_cols].corr().round(decimals=2), annot = True)


# In[ ]:


sns.lineplot(data=train_data, x='WeekOfYear', y='Weekly_Sales')
plt.xlabel('Week')
plt.ylabel('Weekly Sales')
plt.title('Trend of Weekly Sales over Time')
plt.show()


# In[ ]:


avg_sales_by_store = train_data.groupby('Store')['Weekly_Sales'].mean().reset_index()
sns.barplot(data=avg_sales_by_store, x='Store', y='Weekly_Sales', 
            order = avg_sales_by_store.sort_values('Weekly_Sales',ascending = False)['Store'])
plt.xlabel('Store')
plt.ylabel('Average Weekly Sales')
plt.title('Average Weekly Sales by Store')
plt.xticks(rotation=90)  # Rotate X-axis labels by 45 degrees
plt.tight_layout()  # Adjust layout to prevent label overlapping
plt.show()


# In[ ]:


sns.scatterplot(data=train_data, x='Temperature', y='Weekly_Sales')
plt.xlabel('Temperature')
plt.ylabel('Weekly Sales')
plt.title('Relationship between Temperature and Weekly Sales')
plt.show()


# In[ ]:


sns.boxplot(data=train_data, x='Type', y='Weekly_Sales',showfliers = False)
plt.xlabel('Store Type')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Distribution by Store Type')
plt.show()


# In[ ]:


avg_sales_by_month = train_data.groupby('Month')['Weekly_Sales'].mean().reset_index()
sns.barplot(data=avg_sales_by_month, x='Month', y='Weekly_Sales')
plt.xlabel('Month')
plt.ylabel('Average Weekly Sales')
plt.title('Average Weekly Sales by Month')
plt.show()


# In[ ]:


sns.scatterplot(data=train_data, x='Size', y='Weekly_Sales', hue='Type')
plt.xlabel('Store Size')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales vs. Store Size with Store Type')
plt.show()


# In[ ]:


sns.boxplot(data=train_data, x='IsHoliday', y='Weekly_Sales')
plt.xlabel('Is Holiday')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Distribution for Holidays vs. Non-Holidays')
plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
plt.show()


# In[ ]:


avg_sales_by_year = train_data.groupby('Year')['Weekly_Sales'].sum().reset_index()
# Plot the average weekly sales by years
sns.barplot(data=avg_sales_by_year, x='Year', y='Weekly_Sales')
plt.xlabel('Year')
plt.ylabel('Average Weekly Sales')
plt.title('Average Weekly Sales by Years')
plt.show()


# In[ ]:


sns.lineplot(train_data, x = 'WeekOfYear', y = 'Weekly_Sales', hue = 'Year', palette = ['blue', 'red', 'green'])
plt.xlabel('Week Of Year')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales vs. Week of Year')
plt.show()


# In[ ]:


avg_sales_by_store = train_data.sample(1000).groupby(['Dept'])['Weekly_Sales'].mean().reset_index()
plt.figure(figsize = (80,30))
sns.barplot(data=avg_sales_by_store, x='Dept', y='Weekly_Sales', order = avg_sales_by_store.sort_values('Weekly_Sales',ascending = False)['Dept'])
plt.xlabel('Department', fontsize = 100)
plt.ylabel('Average Weekly Sales', fontsize = 100)
plt.title('Average Weekly Sales by Department', fontsize = 100)
plt.yticks(fontsize=60)
plt.xticks(fontsize=60)
plt.xticks(rotation=90)  # Rotate X-axis labels by 45 degrees
plt.tight_layout()  # Adjust layout to prevent label overlapping
plt.show()


# In[ ]:


avg_sales_by_month = train_data.groupby('Quarter')['Weekly_Sales'].mean().reset_index()
sns.barplot(data=avg_sales_by_month, x='Quarter', y='Weekly_Sales')
plt.xlabel('Quarter')
plt.ylabel('Average Weekly Sales')
plt.title('Average Weekly Sales by Quarter')
plt.show()


# In[ ]:


avg_sales_by_month = train_data.groupby(['Day', 'Month'])['Weekly_Sales'].mean().reset_index()
sns.lineplot(data=avg_sales_by_month, x='Day', y='Weekly_Sales', hue = 'Month', palette = sns.color_palette("Paired", 12))
plt.xlabel('Day')
plt.ylabel('Average Weekly Sales')
plt.title('Average Weekly Sales by Day')
plt.show()


# In[ ]:


train_size = int(.75 * len(train_data))
train_df,val_df = train_data[:train_size], train_data[train_size:]


# In[ ]:


len(train_df), len(val_df)


# In[ ]:


input_cols = ['Store', 'Dept', 'IsHoliday', 'Temperature','Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size', 'Day', 'Month','Year', 'WeekOfYear', 'Quarter', 'MarkDown']
target_col = 'Weekly_Sales'


# In[ ]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_data[input_cols].copy()


# In[ ]:


def num_cat_cols(data):
    
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes('object').columns.tolist()
    
    return numeric_cols, categorical_cols


# In[ ]:


numeric_cols, categorical_cols = num_cat_cols(train_inputs)


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols+encoded_cols]
X_val = val_inputs[numeric_cols+encoded_cols]
X_test = test_inputs[numeric_cols+encoded_cols]


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


# In[ ]:


linear_models_scores = {}


# In[ ]:


def try_linear_models(model_name, model):
    
    model.fit(X_train, train_targets) # training the model on training data
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val) # model predictions on validation data
    
    # Training prediction scores
    train_mae = mean_absolute_error(train_targets, train_preds)
    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    train_r2 = r2_score(train_targets, train_preds)
    # validation prediction scores
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    val_r2 = r2_score(val_targets, val_preds)
    
    linear_models_scores[model_name] = {'mae':[round(train_mae,2), round(val_mae,2)],'rmse':[round(train_rmse,2), round(val_rmse,2)],'r2':[round(train_r2,2), round(val_r2,2)]}
    
    return val_mae, val_rmse, val_r2


# In[ ]:


model_names = ['linear', 'ridge', 'lasso', 'elasticnet', 'sgd']
models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), SGDRegressor()]

for i in range(len(models)):
    
    val_mae, val_rmse, val_r2 = try_linear_models(model_names[i], models[i])


# In[ ]:


pd.DataFrame(linear_models_scores)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[ ]:


rf_scores = {}
gb_scores = {}
ab_scores = {}
xgb_scores = {}
lgbm_scores = {}
trained_models = {}


# In[ ]:


def try_ensemble_methods(model_name, model, score_dict):
    
    model.fit(X_train, train_targets.values.ravel())
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Training prediction scores
    train_mae = mean_absolute_error(train_targets, train_preds)
    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    train_r2 = r2_score(train_targets, train_preds)
    # validation prediction scores
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    val_r2 = r2_score(val_targets, val_preds)
    
    score_dict['mae'] = {'training':round(train_mae,2), 'validation': round(val_mae,2)}
    score_dict['rmse'] = {'training':round(train_rmse,2), 'validation': round(val_rmse,2)}
    score_dict['r2'] = {'training':round(train_r2,2), 'validation': round(val_r2,2)}
    
    trained_models[model_name] = model
    
    return val_mae, val_rmse, val_r2


# In[ ]:


ensemble_models = [RandomForestRegressor(random_state=42, n_jobs=-1),
                  GradientBoostingRegressor(random_state=42),
                  AdaBoostRegressor(random_state=42),
                  XGBRegressor(random_state=42, n_jobs=-1),
                  LGBMRegressor(random_state=42, n_jobs=-1)]
ensemble_model_names = ['random_forest','gradient_boosting',
                       'adaboost','xgboost','lightgbm']
score_dicts = [rf_scores,gb_scores, ab_scores, xgb_scores, lgbm_scores]


for i in range(5):
    
    val_mae, val_rmse, val_r2 = try_ensemble_methods(ensemble_model_names[i], ensemble_models[i], score_dicts[i])
    print("*********")
    print(ensemble_model_names[i])
    print("Val MAE: ", val_mae)
    print("Val RMSE: ", val_rmse)
    print("Val R2: ", val_r2)


# In[ ]:


print("Random Forest scores")
display(pd.DataFrame(rf_scores))
print("XGBoost Scores")
display(pd.DataFrame(xgb_scores))
print("LightGBM Scores")
display(pd.DataFrame(lgbm_scores))
print("Gradient Boosting Scores")
display(pd.DataFrame(gb_scores))
print("AdaBoost Scores")
display(pd.DataFrame(ab_scores))


# In[ ]:


rf = trained_models['random_forest']
xgb = trained_models['xgboost']
lgbm = trained_models['lightgbm']


# In[ ]:


rf_imp_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=rf_imp_df.head(10), x='importance', y='feature');


# In[ ]:


xgb_imp_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=xgb_imp_df.head(10), x='importance', y='feature');


# In[ ]:


lgbm_imp_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=lgbm_imp_df.head(10), x='importance', y='feature');


# In[ ]:


def weighted_mean_absolute_error(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# In[ ]:


# Creating the weights
train_weights = np.where(X_train['IsHoliday'] == 1, 5, 1)
val_weights = np.where(X_val['IsHoliday'] == 1, 5, 1)


# In[ ]:


models = {
    'randomforest': {
        'model': RandomForestRegressor,
        'params': {
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'n_estimators': [20, 50, 100, 150, 200, 250, 300,500],
            'min_samples_split': [2, 3, 4, 5, 10]
        }
    },
    'xgboost': {
        'model': XGBRegressor,
        'params': {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'n_estimators': [30, 50, 100, 150, 200, 250, 300, 500],
            'learning_rate': [0.3, 0.2, 0.1, 0.01, 0.001]
        }
    },
    'lightgbm': {
        'model': LGBMRegressor,
        'params': {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'n_estimators': [30, 50, 100, 150, 200, 250, 300, 500],
            'learning_rate': [0.3, 0.2, 0.1, 0.01, 0.001]
        }
    }
}


# In[ ]:


results = {}

def test_params(model_type, model, **params):
    model_instance = model(**params)
    model_instance.fit(X_train, train_targets)
    train_wmae = weighted_mean_absolute_error(model_instance.predict(X_train), train_targets, train_weights)
    val_wmae = weighted_mean_absolute_error(model_instance.predict(X_val), val_targets, val_weights)
    
    return train_wmae, val_wmae

def test_param_and_plot(model_type, model, param_name, param_values):
    
    train_errors, val_errors = [], []
    wmae_results = {}
    
    for value in param_values:
        params = {param_name: value}
        train_wmae, val_wmae = test_params(model_type, model, **params)
        
        train_errors.append(train_wmae)
        val_errors.append(val_wmae)
        
    
    plt.figure(figsize=(10, 6))
    plt.title(model_type + ' Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('WMAE')
    plt.legend(['Training', 'Validation'])
    
    wmae_results[param_name] = {
            'train_wmae': train_errors,
            'val_wmae': val_errors
        }
    
    return wmae_results


# In[ ]:


# Iterate over each model type
for model_type, config in models.items():
    model = config['model']
    params = config['params']
    
    # Iterate over each parameter and its values
    for param_name, param_values in params.items():
        wmae_results = test_param_and_plot(model_type, model, param_name, param_values)
        results[model_type] = wmae_results


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


model = RandomForestRegressor(random_state=42, n_jobs=-1)


# In[ ]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, train_targets)

best_model = grid_search.best_estimator_

scores = cross_val_score(best_model, X_train, train_targets, cv=5, scoring='neg_mean_absolute_error')
wmae_scores = -scores

print("Cross-validation WMAE scores: ", wmae_scores)
print("Mean WMAE: ", np.mean(wmae_scores))

best_model.fit(X_train, train_targets)

print("Validation WMAE")
rf_val_wmae = weighted_mean_absolute_error(val_targets, best_model.predict(X_val), val_weights)


# In[ ]:


rf_val_wmae


# In[ ]:


xgb_model = XGBRegressor(random_state=42, n_jobs=-1)


# In[ ]:


xgb_param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, train_targets)

xgb_best_model = grid_search.best_estimator_

scores = cross_val_score(xgb_best_model, X_train, train_targets, cv=5, scoring='neg_mean_absolute_error')
xgb_mae_scores = -scores
print("Cross-validation MAE scores:", xgb_mae_scores)
print("Mean MAE:", np.mean(xgb_mae_scores))

xgb_best_model.fit(X_train, train_targets)

print("Validation WMAE")
xgb_val_wmae = weighted_mean_absolute_error(val_targets, xgb_best_model.predict(X_val), val_weights)


# In[ ]:


xgb_val_wmae


# In[ ]:


import joblib

walmart_sales = {
    'rf_model':best_model,
    'xgb_model':xgb_best_model,
    'imputer':imputer,
    'scaler':scaler,
    'encoder':encoder,
    'input_cols':input_cols,
    'target_cols':target_col,
    'numeric_cols':numeric_cols,
    'categorical_cols':categorical_cols,
    'encoded_cols':encoded_cols
}


# In[ ]:


joblib.dump(walmart_sales, 'walmart_sales.joblib')


# In[ ]:


test_preds1 = best_model.predict(X_test)
test_preds2 = xgb_best_model.predict(X_test)


# In[ ]:


submission = pd.read_csv("sampleSubmission.csv", header = 0)
submission['Weekly_Sales'] = test_preds2
submission.fillna(0, inplace=True)
submission.to_csv("submission.csv", index=None)


# In[ ]:





# In[ ]:




