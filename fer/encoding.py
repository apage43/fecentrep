import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class IdentifierLabeler:
    def __init__(self, cols):
        self.encoder = LabelEncoder()
        self.cols = cols

    def fit(self, df):
        self.encoder.fit(pd.concat(df[col] for col in self.cols))
        return self

    def transform(self, df):
        for col in self.cols:
            df[col] = self.encoder.transform(df[col])
        return df

    def fit_transform(self, df, **fp):
        self.fit(df)
        return self.transform(df)
