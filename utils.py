import os
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# read csv data

def read_data(binaryClass):
    # data = read_data("AD_CN")
    modalities      = ["MRI","GM", "PET", "CSF", "SNP"]
    
    X_modalities = []
    for modality in modalities:
        df_i    = pd.read_csv(os.path.join("data", binaryClass, f"{modality}.csv"),
                              header=None)
        X_i     = df_i.values   
        print(f"modality: {modality}, shape: {X_i.shape}")
        X_modalities.append(X_i)

    data = {}
    data["X"] = X_modalities
    data["X_label"] = modalities
    le = LabelEncoder()
    df_i  = pd.read_csv(os.path.join("data", binaryClass, f"{binaryClass}_label.csv"))
    y = df_i["Research Group"].values
    data["y_label"] = y
    y_encoded = le.fit_transform(y)
    data["y"] = y_encoded
    
    return data
