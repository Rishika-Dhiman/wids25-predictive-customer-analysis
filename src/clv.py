import numpy as np

def compute_clv(rfm, churn_p):
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']

    rfm['clv'] = (rfm['avg_order_value'] * rfm['frequency'] * (1 / (churn_p + 1e-6)))

    return rfm