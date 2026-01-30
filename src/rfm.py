import pandas as pd
from datetime import timedelta

def compute_rfm(df):
    last_date = df['InvoiceDate'].max() + timedelta(days = 1)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (last_date - x.max()).days,
        'Invoice': 'nunique',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    rfm['R'] = pd.qcut(rfm['recency'], 4, labels = [4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['frequency'], 4, labels = [1, 2, 3, 4])
    rfm['M'] = pd.qcut(rfm['monetary'], 4, labels = [1, 2, 3, 4])
    rfm['RFM_score'] = rfm[['R', 'F', 'M']].astype(int).sum(axis = 1)

    return rfm