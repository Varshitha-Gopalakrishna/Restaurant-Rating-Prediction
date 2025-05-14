import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import dill

def train_and_compare_models(df):
    X = df[['online_order', 'book_table', 'votes', 'location', 'rest_type', 'cuisines', 'approx_cost_for_2_people']]
    y = df['rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    rf_preds = best_rf.predict(X_test)
    results['RandomForest'] = {
        'model': best_rf,
        'r2': r2_score(y_test, rf_preds),
        'mae': mean_absolute_error(y_test, rf_preds)
    }

    # XGBoost
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    xgb_preds = best_xgb.predict(X_test)
    results['XGBoost'] = {
        'model': best_xgb,
        'r2': r2_score(y_test, xgb_preds),
        'mae': mean_absolute_error(y_test, xgb_preds)
    }

    # CatBoost
    cat = CatBoostRegressor(verbose=0, random_state=42)
    cat.fit(X_train, y_train)
    cat_preds = cat.predict(X_test)
    results['CatBoost'] = {
        'model': cat,
        'r2': r2_score(y_test, cat_preds),
        'mae': mean_absolute_error(y_test, cat_preds)
    }

    # Print results
    for name, metrics in results.items():
        print(f"📊 {name:<12} → R2: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")

    # Choose best model
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_model = results[best_name]['model']

    # Save best model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    if best_name == 'XGBoost':
        model_path = os.path.join(model_dir, 'model.xgb')
        best_model.save_model(model_path)  # Native saving
    else:
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, "wb") as f:
            dill.dump(best_model, f)

    print(f"✅ Saved best model ({best_name}) to {model_path}")
