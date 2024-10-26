
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import model_ops

## Calculate performance metrics 
from sklearn.metrics import mean_absolute_error, r2_score

def model_evaluation(tD_results):
    mae = mean_absolute_error(tD_results["y_test"], tD_results["y_pred"])
    r2 = r2_score(tD_results["y_test"], tD_results["y_pred"])
    return {"MSE": mae, "r2": r2}



def linear_regression(tD_datasets, strURL):
    tModel = LinearRegression()
    tModel.fit(tD_datasets["X_train"], tD_datasets["y_train"])

    tD_datasets["y_pred"] = tModel.predict(tD_datasets["X_test"])
    tD_results = model_evaluation(tD_datasets)
    model_ops.save(tModel, strURL)
    return {"model": tModel, "evaluation_results": tD_results}

