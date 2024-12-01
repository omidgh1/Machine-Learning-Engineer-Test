import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self, outlier_factor=1.5, imputation_strategy='mean', verbose=True):
        """
        Data Preprocessor for feature engineering, outlier handling, and scaling.
        :param outlier_factor: Multiplier for the IQR in outlier detection.
        :param imputation_strategy: Strategy for filling missing values ('mean' or 'median').
        :param verbose: If True, enable logging for debugging.
        """
        self.outlier_factor = outlier_factor
        self.imputation_strategy = imputation_strategy
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.stats = {}
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def fit(self, X: pd.DataFrame):
        """
        Fit the preprocessor by calculating statistics for the given data.
        """
        self.quantitative = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        if self.verbose:
            logging.info("Fitting preprocessor...")

        self.stats['quantiles'] = {
            col: {
                'q1': X[col].quantile(0.25),
                'q3': X[col].quantile(0.75)
            }
            for col in self.quantitative
        }
        self.stats['means'] = X[self.quantitative].mean()
        self.stats['medians'] = X[self.quantitative].median()
        self.stats['lat_mean'] = X['latitude'].mean()
        self.stats['lon_mean'] = X['longitude'].mean()

    def handle_outliers(self, X: pd.DataFrame):
        """
        Handle outliers by capping values using the IQR method.
        """
        if self.verbose:
            logging.info("Handling outliers...")

        self.quantitative = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        for col in self.quantitative:
            q1 = self.stats['quantiles'][col]['q1']
            q3 = self.stats['quantiles'][col]['q3']
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_factor * iqr
            upper_bound = q3 + self.outlier_factor * iqr
            X[col] = np.clip(X[col], lower_bound, upper_bound)

    def fill_missing(self, X: pd.DataFrame):
        """
        Fill missing values using the specified imputation strategy.
        """
        if self.verbose:
            logging.info("Filling missing values...")

        for col in self.quantitative:
            value = self.stats['means'][col] if self.imputation_strategy == 'mean' else self.stats['medians'][col]
            X[col] = X[col].fillna(value)

    def create_features(self, X: pd.DataFrame):
        """
        Create new features for the dataset.
        """
        if self.verbose:
            logging.info("Creating new features...")

        X['population_per_room'] = X['population'] / X['total_rooms']
        X['bedroom_share'] = X['total_bedrooms'] / X['total_rooms'] * 100
        X['diag_coord'] = X['longitude'] + X['latitude']
        X['distance_to_center'] = self.haversine(X['latitude'], X['longitude'], self.stats['lat_mean'],
                                                 self.stats['lon_mean'])
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['income_per_population'] = X['median_income'] * X['population']

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points.
        """
        R = 6371.0
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def encode_categorical(self, X: pd.DataFrame):
        """
        Encode categorical columns.
        """
        if self.verbose:
            logging.info("Encoding categorical features...")

        if 'grid' in X:
            X['grid_encoded'] = self.label_encoder.fit_transform(X['grid'])
            X = X.drop(columns=['grid'])

    def scale_features(self, X: pd.DataFrame):
        """
        Scale numerical features.
        """
        if self.verbose:
            logging.info("Scaling features...")

        X[self.quantitative] = self.scaler.fit_transform(X[self.quantitative])

    def transform(self, X: pd.DataFrame):
        """
        Apply all transformations to the dataset.
        """
        X = X.copy()
        self.handle_outliers(X)
        self.fill_missing(X)
        self.create_features(X)
        self.encode_categorical(X)
        self.scale_features(X)
        del X['ocean_proximity']
        return X