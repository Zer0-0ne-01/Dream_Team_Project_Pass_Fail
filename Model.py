import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def pred(train_addres="./student-por.csv", data_addres="./data.csv"):
    df = pd.read_csv(train_addres, sep=";")
    target = (df['G3'] < 10).astype(int)
    df = df.drop(columns="G3")
    df = pd.get_dummies(df, drop_first=True)
    
    rf = RandomForestClassifier(n_estimators=1000, max_depth=10,
                                class_weight='balanced', random_state=42)
    rf.fit(df, target)
    
    p_df = pd.read_excel(data_addres)
    if 'G3' in p_df.columns:
        p_df = p_df.drop(columns='G3')
    p_df = pd.get_dummies(p_df, drop_first=True)
    p_df = p_df.reindex(columns=df.columns, fill_value=0)
    
    threshold = 0.61
    proba = rf.predict_proba(p_df)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred


