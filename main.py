import pandas as pd
from src.clv import compute_clv
from src.rfm import compute_rfm
from src.churn import label_churn, train_churn_model

df = pd.read_csv('data/online_retail_ll.csv')

df = df.dropna(subset = ['Customer ID'])

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Amount'] = df['Quantity'] * df['Price']
df = df[df['Amount'] > 0]

rfm = compute_rfm(df)
rfm = label_churn(rfm)
lr, rf, X_test, y_test = train_churn_model(rfm)

churn_p = rf.predict_proba(
    rfm[['recency', 'frequency', 'monetary', 'RFM_score']]
)
churn_p = churn_p[:, 1]

rfm = compute_clv(rfm, churn_p)
rfm.to_csv('data/result_rfm_clv_churn.csv', index = False)
