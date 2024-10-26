import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def get_data_online():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return [data, target]

def print_data_report(mReport, mTitle):
    print(mTitle)
    print(mReport)
    print("*")
          
def data_review(df_data):
    # View the first few rows of the dataset
    tReport = df_data.head()
    print_data_report(tReport, "**** Data Format ****")

    # Check for missing values
    tReport = df_data.isnull().sum()
    print_data_report(tReport, "**** Summary of Missing Values ****")

    # Statistical summary of numerical columns
    tReport = df_data.describe()
    print_data_report(tReport, "**** Statistics ****")

def get_data_locally():
    strFileName = "boston.csv"
    raw_df = pd.read_csv(strFileName)
    df_data = pd.DataFrame(raw_df)

    data_review(df_data)
    
    return df_data

def plot_corr(corr_matrix, strPathFile):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    #plt.show()
    plt.savefig(strPathFile)
    print("The correlation image is saved in " + strPathFile + ".")

def plot_correlation(df):
    strPath = "./results/"
    strFile = "plot_corr_matrix.png"

    corr_matrix = df.corr()
    plot_corr(corr_matrix, strPath + strFile)

    column_names = list(df.columns)
    if "MEDV" in column_names:
        print("MEDV")
    strItem = column_names[0]
    tCorr = corr_matrix["MEDV"]
    tListNames =list()
    tListValues = list()
    for tD in column_names:
        tValue1 = 0
        tName1 = ""
        for tName0 in column_names:
            tValue0 = abs(float(tCorr[tName0]))
            if tValue0 > tValue1 and tName0 not in tListNames:
                tValue1 = tValue0
                tName1 = tName0
        tListNames.append(tName1)
        tListValues.append(tValue1)

    #df[tListNames] = df[column_names]
    df_sorted = df.copy()

    tD = {}
    for i in range(len(tListNames)):
        tD[column_names[i]] = tListNames[i]
    df_sorted.rename(columns=tD, inplace=True)

    for tName in column_names:
        df_sorted[tName] = df[tName]

    corr_matrix = df_sorted.corr()
    corr_matrix = abs(corr_matrix)
    column_names1 = list(df_sorted.columns)
    strFile = "plot_corr_matrix_sorted.png"
    strPathFile = strPath + strFile
    plot_corr(corr_matrix, strPathFile)
    

def plot_histogram(df, strFile):
    df.hist(bins=30, figsize=(20,15))
    #plt.show()
    sns.pairplot(df, diag_kind='hist', corner=True)
    #plt.show()
    plt.savefig(strFile)
    print("The histogram image is saved in " + strFile + ".")

def remove_outliers(df_data, strVariable):
    #To enhance the robustness of the model, outliers in the target variable are addressed. The dataset is filtered to include only those instances where the sale price falls between the 5th and 95th percentiles.
    #df = df.drop(df[df.score < 50].index)
    df_data.sort_values(by=[strVariable], ascending=True)
    tNumOfRows = len(df_data)
    tLimitLow = int(tNumOfRows*0.05)
    tLimitHigh = int(tNumOfRows*0.95)
    tDF = df_data.iloc[tLimitLow:tLimitHigh, :]
    print("Removed outliers for " + strVariable + " (n = " + str(len(df_data) - len(tDF)) + ").")
    return tDF

def remove_non_digit(df_data, strVariable):
    tDF = df_data[df_data[strVariable].apply(lambda x: str(x).isdigit())]
    print("Removed non-digit values for " + strVariable + " (n = " + str(len(df_data) - len(tDF)) + ").")
    return tDF

def get_df_training_datasets(df, strVariableToBePredicted):
    X = df.drop(strVariableToBePredicted, axis=1)
    y = df[strVariableToBePredicted]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

