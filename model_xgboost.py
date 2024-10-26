from xgboost import XGBRegressor
import model_evaluation

import json

def save_json(tD_data, strPathFile):
    # Serialize data into file:
    json.dump( tD_data, open( strPathFile, 'w' ) )

def load_json(strPathFile):
    # Read data from file:
    tD_data = json.load( open( "file_name.json" ) )
    return tD_data

def train(tD_Datasets):
    tRange_n_estimators = 64
    tRange_max_depth = 8
    learning_rate_array = [1, 0.5, 0.25, 0.125]
    tMSE_Lowest = None
    tMode_Settings = {}
    tD_data = {}
    tD_data["xpos"] = list()
    tD_data["ypos"] = list()
    tD_data["dz"] = list()
    tD_Results = {}
    tD_Results["y_true"] = tD_Datasets["y_test"]

    tBoolUpdatePos = True
    for t_learning_rate in learning_rate_array:
        tListDZ = list()
        for i in range(tRange_n_estimators):
            for j in range(tRange_max_depth):
                t_n_estimators = i*2 +2
                t_max_depth = j+2
                if tBoolUpdatePos== True:
                    tD_data["xpos"].append(t_n_estimators)
                    tD_data["ypos"].append(t_max_depth)

                xgb = XGBRegressor(n_estimators=t_n_estimators, max_depth=t_max_depth, learning_rate=t_learning_rate)
                #xgb = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
                # fit model
                xgb.fit(tD_Datasets["X_train"], tD_Datasets["y_train"])
                # make predictions
                preds = xgb.predict(tD_Datasets["X_test"])
                tMSE = model_evaluation.calculate_mse(tD_Results["y_true"], preds)
                tListDZ.append(tMSE)
                tBoolUpdate = False
                if tMSE_Lowest is None:
                    tBoolUpdate = True
                else:
                    if tMSE<tMSE_Lowest:
                        tBoolUpdate = True
                if tBoolUpdate == True:
                    tMSE_Lowest = tMSE
                    tMode_Settings["n_estimators"] = t_n_estimators
                    tMode_Settings["max_depth"] = t_max_depth
                    tMode_Settings["learning_rate"] = t_learning_rate
                    tD_Results["y_pred"] =preds
        tD_data["dz"].append(tListDZ)
        tBoolUpdatePos = False

    tD_data["xlabel"] = "n_estimators"
    tD_data["ylabel"] = "t_max_depth"
    tD_data["zlabel"] = "MSE"
    tD_data["title"] = "XGBoost Performance with Different Settings"
    tD_Results = {"optimal_settings": tMode_Settings, "the_lowest_mse": tMSE_Lowest, "data_of_bars": tD_data}

    strPath = "./results/"
    strFile = "xgboost_training_results.txt"
    save_json(tD_Results, strPath + strFile)

    return tD_Results

def evaluation(tD_Results):
    tMSE = model_evaluation.calculate_mse(tD_Results)
    tR2 = model_evaluation.calculate_r2(tD_Results)
    return {"MSE": tMSE, "R2": tR2}


