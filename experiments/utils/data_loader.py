import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path):

    print(f"[DATA] Loading: {path}")
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        np.array(X_train, dtype=float),
        np.array(y_train, dtype=float),
        np.array(X_test, dtype=float),
        np.array(y_test, dtype=float),
    )
