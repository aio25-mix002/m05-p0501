from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer,  IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import mutual_info_regression, SelectKBest, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder, PolynomialFeatures, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
import streamlit as st
from src.app.startup import GLOBAL_SEED, TEST_SIZE


def processing_pipeline(df, val_set=False, polynomial=True):
    ''' Processing pipeline '''
    df = drop_missing(df)
    num_cols = [col for col in df.columns if df[col].dtype in ["float64","int64"]]
    cat_cols = [col for col in df.columns if df[col].dtype not in ["float64","int64"]]
    num_cols.remove('SalePrice')
    
    # imputer = SimpleImputer(strategy='median')
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    scaler = MinMaxScaler()
    # 1. Data Splitting
    if val_set:
        train_df, test_df, val_df = data_split(df, val_set=val_set)
        y_train, y_test, y_val = train_df['SalePrice'].values, test_df['SalePrice'].values, val_df['SalePrice'].values
        train_df = train_df.drop(["SalePrice"], axis=1), 
        test_df = test_df.drop(["SalePrice"], axis=1), 
        val_df = val_df.drop(["SalePrice"], axis=1)
        
        # Missing imputation
        train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
        test_df[num_cols], val_df[num_cols] = num_imputer.transform(test_df[num_cols]), num_imputer.transform(val_df[num_cols])
        
        train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])
        val_df[cat_cols] = cat_imputer.transform(val_df[cat_cols])
        
        # One-hot encoding for categorical features
        train_df_encoded, encoder = one_hot_encode(train_df, cat_cols)
        test_df_encoded, _ = one_hot_encode(test_df, cat_cols, encoder=encoder)
        val_df_encoded, _ = one_hot_encode(val_df, cat_cols, encoder=encoder)
        
        # Scale numeric data
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols], val_df[num_cols] = scaler.transform(test_df[num_cols]), scaler.transform(val_df[num_cols])
        
        # Make polynomial features
        if polynomial:
            poly_features = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False)
            train_poly_features = poly_features.fit_transform(train_df[num_cols]) 
            test_poly_features = poly_features.transform(test_df[num_cols])
            val_poly_features = poly_features.transform(val_df[num_cols])
            
            X_train = np.hstack([train_poly_features, train_df_encoded])
            X_test = np.hstack([test_poly_features, test_df_encoded])
            X_val = np.hstack([val_poly_features, val_df_encoded])

        else:
            # X_train = stack_features(train_df, num_cols, train_encoded_cols)
            # X_test = stack_features(test_df, num_cols, test_encoded_cols)
            # X_val = stack_features(val_df, num_cols, val_encoded_cols)
            # Combine scaled numerical features and encoded categorical features
            train_combined = pd.concat([train_df[num_cols], train_df_encoded], axis=1)
            test_combined = pd.concat([test_df[num_cols], test_df_encoded], axis=1)
            
            # Align columns
            train_cols = train_combined.columns
            test_combined = test_combined.reindex(columns=train_cols, fill_value=0)

            X_train = train_combined.values
            X_test = test_combined.values
            
        return (X_train, y_train), (X_test, y_test), (X_val, y_val)
    
    else:
        train_df, test_df = data_split(df, val_set=val_set)
        y_train, y_test = train_df['SalePrice'].values, test_df['SalePrice'].values
        train_df = train_df.drop(["SalePrice"], axis=1)
        test_df = test_df.drop(["SalePrice"], axis=1)

        
        # Missing imputation
        train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
        test_df[num_cols] = num_imputer.transform(test_df[num_cols])
        
        train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])
        
        # One-hot encoding
        train_df_encoded, encoder = one_hot_encode(train_df, cat_cols)
        test_df_encoded, _ = one_hot_encode(test_df, cat_cols, encoder=encoder)  # handling cases where the train and test sets might have different columns after encoding 
        
        # Scale numeric data
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
    
        # Make polynomial features
        if polynomial:
            poly_features = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False)
            train_poly_features = poly_features.fit_transform(train_df[num_cols])
            test_poly_features = poly_features.transform(test_df[num_cols])
            
            # Align columns after one-hot encoding to ensure they match
            train_cols = train_df_encoded.columns
            test_df_encoded = test_df_encoded.reindex(columns=train_cols, fill_value=0)
        
            X_train = np.hstack([train_poly_features, train_df_encoded])
            X_test = np.hstack([test_poly_features, test_df_encoded])

        else:
            # X_train = stack_features(train_df, num_cols, encoder.get_feature_names_out(cat_cols))
            # X_test = stack_features(test_df, num_cols, encoder.get_feature_names_out(cat_cols))
            
            # Combine scaled numerical features and encoded categorical features
            train_combined = pd.concat([train_df[num_cols], train_df_encoded], axis=1)
            test_combined = pd.concat([test_df[num_cols], test_df_encoded], axis=1)
            val_combined = pd.concat([val_df[num_cols], val_df_encoded], axis=1)
            
            # Align columns
            train_cols = train_combined.columns
            test_combined = test_combined.reindex(columns=train_cols, fill_value=0)
            val_combined = val_combined.reindex(columns=train_cols, fill_value=0)

            X_train = train_combined.values
            X_test = test_combined.values
            X_val = val_combined.values
        
        return (X_train, y_train), (X_test, y_test)


def drop_missing(df):
    ''' Drop features with > 50% missing '''
    return df.drop(["Id","Alley","PoolQC","Fence","MiscFeature"], axis=1)

def data_split(df, test_size= TEST_SIZE, val_set=False, bins= False):
    ''' Split train and test set with stratifying on SalePrice. Make validation set if needed '''
    # Create bins based on SalePrice quantiles to ensure balanced representation
    if bins == False:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
        )
        return train_val_df, test_df
    
    n_bins = 5 
    df_copy = df.copy()
    df_copy['SalePrice_bins'] = pd.qcut(df_copy['SalePrice'], 
                                    q=n_bins, 
                                    labels=False, 
                                    duplicates='drop')

    # Stratified split
    train_val_df, test_df = train_test_split(
        df_copy,
        test_size=0.25,
        random_state=42,
        stratify=df_copy['SalePrice_bins']  # Stratify based on price bins
    )

    
    if val_set:
        train_df, val_df = train_test_split(train_val_df, 
                                            test_size=0.25, 
                                            random_state=42,
                                            stratify=train_val_df['SalePrice_bins']
                                            )
        
        # Remove the temporary binning column
        train_df = train_df.drop('SalePrice_bins', axis=1)
        test_df = test_df.drop('SalePrice_bins', axis=1)
        val_df = val_df.drop('SalePrice_bins', axis=1)
        
        return train_df, test_df, val_df
    
    # Remove the temporary binning column
    train_df = train_val_df.drop('SalePrice_bins', axis=1)
    test_df = test_df.drop('SalePrice_bins', axis=1)
    
    return train_df, test_df
    
def one_hot_encode(df, cat_cols, encoder=None):
    ''' One-hot encode categorical columns '''
    df_encoded = df.copy()
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity,
        encoder.fit(df_encoded[cat_cols].astype(str))

    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    df_encoded[encoded_cols] = encoder.transform(df_encoded[cat_cols].astype(str))
    df_encoded = df_encoded.drop(cat_cols, axis=1)
    return df_encoded, encoder

def stack_features(df, num_cols, encoded_cols):
    ''' Stack numerical and encoded columns '''
    return np.hstack([df[num_cols], df[encoded_cols]])

def preprocessing(df, add_features=False):
    df = drop_missing(df)
    if add_features:
        df = create_new_feature(df)
    else:
        corr_matrix = df.corr(numeric_only=True).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        df = df.drop(columns=to_drop)
    num_cols = [col for col in df.columns if df[col].dtype in ["float64","int64"]]
    cat_cols = [col for col in df.columns if df[col].dtype not in ["float64","int64"]]
    num_cols.remove('SalePrice')
    train_df, test_df = data_split(df)
    X_train, y_train = train_df.drop(columns='SalePrice'), train_df['SalePrice']
    X_test, y_test = test_df.drop(columns='SalePrice'), test_df['SalePrice']
    return X_train, y_train, X_test, y_test, num_cols, cat_cols

def create_pipe(num_cols, num_imp, num_tran, num_scal, cat_cols, feature_name,k):
    if num_imp == 'KNN Imputer':
        num_proc = Pipeline([('imp',KNNImputer())])
    elif num_imp == 'Iterative Imputer':
        num_proc = Pipeline([('imp',IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=GLOBAL_SEED))])
    else:
        num_proc = Pipeline([('imp',SimpleImputer(strategy='median'))])

    #if add_feature:
        #num_proc.steps.append(('add',FunctionTransformer(add_new_features,validate=False)))
    
    if num_tran != 'Yeo-Johnson':
        num_proc.steps.append(('trans',PowerTransformer(method='yeo-johnson', standardize=True)))
    elif num_tran == 'Quantile':
        num_proc.steps.append(('trans',QuantileTransformer(output_distribution='normal', random_state=GLOBAL_SEED)))

    if num_scal == 'Robust Scaler':
        num_proc.steps.append(('sc',RobustScaler()))
    elif num_scal == 'MinMaxScaler':
        num_proc.steps.append(('sc',MinMaxScaler()))
    elif num_scal == 'StandardScaler':
        num_proc.steps.append(('sc',StandardScaler()))
    
    cat_proc = Pipeline([('imp',SimpleImputer(strategy = 'most_frequent')),('ohe',OneHotEncoder(handle_unknown = 'ignore', sparse_output = False,drop='first'))])
    

    preprocess = ColumnTransformer([('num',num_proc,num_cols),('cat',cat_proc,cat_cols)], verbose_feature_names_out = False)
    pipe = Pipeline([("preprocessor", preprocess)])


    if feature_name == 'Random Forest':
        pipe.steps.append(("select", SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=GLOBAL_SEED),threshold=-np.inf, max_features = k)))
    if feature_name == 'XGBoost':
        pipe.steps.append(("select", SelectFromModel(xgb.XGBRegressor(n_estimators=100, random_state=GLOBAL_SEED),threshold=0.01, max_features = k)))
    elif feature_name == 'Mutual Info':
        pipe.steps.append(("select", SelectKBest(score_func=mutual_info_regression, k=k)))

    return pipe

def create_new_feature(df):
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    # Total bathrooms (full + 0.5*half) including basement
    if set(["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]).issubset(df.columns):
        df["TotalBaths"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )

    # Total square footage (basement + floors)
    if set(["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]).issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)

    # Porches
    for col in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if set(["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]).issubset(df.columns):
        df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

    # Binary presence flags
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype("int32")
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype("int32")
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype("int32")
    if "TotalPorchSF" in df.columns:
        df["HasPorch"] = (df["TotalPorchSF"].fillna(0) > 0).astype("int32")

    # Age-related features
    for col in ["YearBuilt", "YearRemodAdd", "YrSold"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if set(["YrSold", "YearBuilt"]).issubset(df.columns):
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    if set(["YrSold", "YearRemodAdd"]).issubset(df.columns):
        df["RemodelAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    if set(["YearBuilt", "YearRemodAdd"]).issubset(df.columns):
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype("int32")
    

    # Simple interaction: quality-weighted living area
    if set(["OverallQual", "GrLivArea"]).issubset(df.columns):
        df["QualXGrLivArea"] = df["OverallQual"].fillna(0) * df["GrLivArea"].fillna(0)
    
    to_drop.extend(["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath","TotalBsmtSF", "1stFlrSF", "2ndFlrSF","OpenPorchSF",
                           "EnclosedPorch", "3SsnPorch", "ScreenPorch","OverallQual", "GrLivArea","YearBuilt", "YearRemodAdd", 
                           "YrSold","PoolArea","GarageArea","Fireplaces","TotalPorchSF"])

    df = df.drop(columns=to_drop)

    return df.copy()