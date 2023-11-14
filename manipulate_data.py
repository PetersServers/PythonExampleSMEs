import pandas as pd
import numpy as np

# Load the DataFrame from your local path
payments_path = 'olist_order_payments_dataset.csv'  # Replace with your local path
order_payments = pd.read_csv(payments_path)

# Define a function that assigns a probability that increases with the number of installments
def calculate_probability(installments):
    # For simplicity, we're using a direct proportion here, you might want to use a different function
    return min(installments / 20, 1)

# Add the 'failed_total_payment' column with probabilities
order_payments['failed_total_payment'] = order_payments['payment_installments'].apply(calculate_probability) > np.random.rand(len(order_payments))

# Save the updated DataFrame back to a CSV in the original directory
order_payments.to_csv(payments_path, index=False)
