from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def processing_pipeline(df, val_set=False, polynomial=True):
    ''' Processing pipeline '''
    df = drop_missing(df)
    num_cols = [col for col in df.columns if df[col].dtype in ["float64","int64"]]
    cat_cols = [col for col in df.columns if df[col].dtype not in ["float64","int64"]]
    
    df, encoded_cols = one_hot_encode(df, cat_cols)
    # df = impute_missing(df, num_cols)
    # df = normalize(df, num_cols)
    df = stack_features(df, num_cols, cat_cols) # Stack numerical and encoded columns
    
    if val_set:
        train_df, test_df, val_df = data_split(df, val_set=val_set)
        y_train, y_test, y_val = train_df['SalePrice'].values, test_df['SalePrice'].values, val_df['SalePrice'].values
        train_df = train_df.drop(["SalePrice"], axis=1)
        test_df = test_df.drop(["SalePrice"], axis=1)
        val_df = val_df.drop(["SalePrice"], axis=1)
        
        # Missing imputation
        imputer = SimpleImputer()
        train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
        test_df[num_cols], val_df[num_cols] = imputer.transform(test_df[num_cols]), imputer.transform(val_df[num_cols])
        
        # Scale numeric data
        scaler = MinMaxScaler()
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols], val_df[num_cols] = scaler.transform(test_df[num_cols]), scaler.transform(val_df[num_cols])
        
        # Make polynomial features
        if polynomial:
            poly_features = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False)
            train_poly_features = poly_features.fit_transform(train_df[num_cols])  # ***Your code here***
            test_poly_features = poly_features.transform(test_df[num_cols])
            val_poly_features = poly_features.transform(val_df[num_cols])
            
            X_train = np.hstack([train_poly_features, train_df[encoded_cols].values])
            X_test = np.hstack([test_poly_features, test_df[encoded_cols].values])
            X_val = np.hstack([val_poly_features, val_df[encoded_cols].values])

        else:
            X_train = stack_feature(train_df, num_cols, encoded_cols)
            X_test = stack_feature(test_df, num_cols, encoded_cols)
            X_val = stack_feature(val_df, num_cols, encoded_cols)
        
        return (X_train, y_train), (X_test, y_test), (X_val, y_val)
    
    else:
        train_df, test_df = data_split(df, val_set=val_set)
        y_train, y_test = train_df['SalePrice'].values, test_df['SalePrice'].values
        train_df = train_df.drop(["SalePrice"], axis=1)
        test_df = test_df.drop(["SalePrice"], axis=1)
        
        # Missing imputation
        imputer = SimpleImputer()
        train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
        test_df[num_cols] = imputer.transform(test_df[num_cols])
   
        
        # Scale numeric data
        scaler = MinMaxScaler()
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        
        train_df = stack_feature(train_df, num_cols, encoded_cols)
        test_df = stack_feature(test_df, num_cols, encoded_cols)
        
        if polynomial:
            poly_features = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False)
            train_poly_features = poly_features.fit_transform(train_df[num_cols])  # ***Your code here***
            test_poly_features = poly_features.transform(test_df[num_cols])
            
            X_train = np.hstack([train_poly_features, train_df[encoded_cols].values])
            X_test = np.hstack([test_poly_features, test_df[encoded_cols].values])

        else:
            X_train = stack_feature(train_df, num_cols, encoded_cols)
            X_test = stack_feature(test_df, num_cols, encoded_cols)

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
    train_df = train_df.drop('SalePrice_bins', axis=1)
    test_df = test_df.drop('SalePrice_bins', axis=1)
    
    return train_df, test_df
    
def one_hot_encode(df, cat_cols):
    ''' One-hot encode categorical columns '''
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df[cat_cols])
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))
    df[encoded_cols] = encoder.transform(df[cat_cols])
    return df, encoded_cols

def stack_features(df, num_cols, encoded_cols):
    ''' Stack numerical and encoded columns '''
    return np.hstack([df[num_cols], df[encoded_cols]])

