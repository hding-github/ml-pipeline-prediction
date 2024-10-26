from google.cloud import aiplatform
import vertexai.preview


@component(
    packages_to_install=["pandas", "pyarrow",  "sklearn"],
    base_image="python:3.9",
    output_component_file="get_wine_data.yaml")

def get_wine_data(
    url: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
    kpi_ouput: Output[Metrics])-> NamedTuple("output", [("train", int),("test",int)]):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split as tts
    
    df_wine = pd.read_csv(url, delimiter=";")
    df_wine['best_quality'] = [ 1 if x>=7 else 0 for x in df_wine.quality] 
    df_wine['target'] = df_wine.best_quality
    df_wine = df_wine.drop(['quality', 'total sulfur dioxide', 'best_quality'], axis=1)
   
   
    train, test = tts(df_wine, test_size=0.3)
    train.to_csv(dataset_train.path + ".csv" , index=False, encoding='utf-8-sig')
    test.to_csv(dataset_test.path + ".csv" , index=False, encoding='utf-8-sig')
    kpi_ouput.log_metric("train_data_size", int(train.shape[0]))
    kpi_ouput.log_metric("test_data_size", int(test.shape[0]))
    return ((train.shape[0]),(test.shape[0]))