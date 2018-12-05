import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


mortality = pd.read_csv("./data/subject_mortality_table.csv", dtype="int")

drug = pd.read_csv("./data/subject_drug_table.csv", dtype="int")

personal_information = pd.read_csv("./data/personal_information.csv", dtype="int")


mortality_detail = pd.merge(personal_information, mortality, on="SUBJECT_ID")
drug_detail = pd.merge(personal_information, drug, on="SUBJECT_ID")

# reduce total data size to 30000
np.random.seed(42)
drop_indices = np.random.choice(30760, 760, replace=False)
mortality_detail = mortality_detail.drop(drop_indices)
drug_detail = drug_detail.drop(drop_indices)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(drug_detail, mortality_detail, test_size=8000.0/30000, random_state=42)
# make holdout set
X_train, X_shared, Y_train, Y_shared = train_test_split(X_train, Y_train, test_size=2000.0/22000, random_state=42)

X_train = X_train.iloc[:, 5:]
X_test = X_test.iloc[:, 5:]
X_shared = X_shared.iloc[:, 5:]
Y_train = Y_train.iloc[:, 5]
Y_test = Y_test.iloc[:, 5]
Y_shared = Y_shared.iloc[:, 5]

# shuffle training set
X_train = X_train.sample(frac=1, random_state=42)
Y_train = Y_train.sample(frac=1, random_state=42)

pd.DataFrame(X_train).to_csv("./IID_data/X_train.csv", header=False, index=False)
pd.DataFrame(X_test).to_csv("./IID_data/X_test.csv", header=False, index=False)
pd.DataFrame(X_shared).to_csv("./IID_data/X_shared.csv", header=False, index=False)
pd.DataFrame(Y_train).to_csv("./IID_data/Y_train.csv", header=False, index=False)
pd.DataFrame(Y_test).to_csv("./IID_data/Y_test.csv", header=False, index=False)
pd.DataFrame(Y_shared).to_csv("./IID_data/Y_shared.csv", header=False, index=False)
