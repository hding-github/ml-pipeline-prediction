import sub_analysis
import sub_data
import model_xgboost
import model_evaluation

strPath = "./results/"
strFile = "plot_histogram.png"

df_data = sub_data.get_data_locally()

#sub_data.plot_histogram(df, strPath + strFile)
#sub_data.plot_correlation(df)

tD_Datasets = sub_data.get_df_training_datasets(df_data, "MEDV")

#strFile = "model_regression.pickle"
#strPathFile = strPath + strFile
# tModel = sub_analysis.linear_regression(tD_Datasets, strPathFile)

tD_Results = model_xgboost.train(tD_Datasets)
model_evaluation.plot_model_performance_3d(tD_Results["data_of_bars"])

print("completed")
