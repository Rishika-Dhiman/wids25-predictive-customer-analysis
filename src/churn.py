from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def label_churn(rfm, thereshold = 45):
    rfm['churn'] = (rfm['recency'] > thereshold).astype(int)
    return rfm

def train_churn_model(rfm):
    X = rfm[['recency', 'frequency', 'monetary', 'RFM_score']]
    y = rfm['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
        )

    lr = LogisticRegression(max_iter = 1000)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    rf.fit(X_train, y_train)

    return lr, rf, X_test, y_test