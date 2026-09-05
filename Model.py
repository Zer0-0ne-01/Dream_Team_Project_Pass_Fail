import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def pred(train_addres = "./student-por.csv", data_addres = "./data.csv"):
    df = pd.read_csv(train_addres,sep=";")

    target = (df['G3'] < 10).astype(int)
    df = df.drop(columns="G3")

    #one hot encoding
    df = pd.get_dummies(df,drop_first=True)
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
    )

    rf.fit(df,target)

    p_df = pd.read_excel(data_addres)
    p_df = pd.get_dummies(df,drop_first=True)

    threshold = 0.61
    proba =rf.predict(p_df)

    pred = (proba >= threshold).astype(int)
    return(pred)



