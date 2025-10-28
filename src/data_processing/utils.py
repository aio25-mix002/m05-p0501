from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import streamlit as st



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
        # train_df = train_df.drop(["SalePrice"], axis=1)
        # test_df = test_df.drop(["SalePrice"], axis=1)

        
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

def data_split(df, val_set=False):
    ''' Split train and test set with stratifying on SalePrice. Make validation set if needed '''
    # Create bins based on SalePrice quantiles to ensure balanced representation
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
