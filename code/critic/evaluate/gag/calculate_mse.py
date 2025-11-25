import numpy as np


def mse_df(df_true, df_pred):
    mse = np.sqrt(((df_true - df_pred) ** 2).mean().mean())
    return mse



