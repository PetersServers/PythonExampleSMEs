import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LinReg_model import PaymentMethodPredictor
from BasicNeuralNet import PaymentMethodNNPredictor

# Load the order payments dataset from the specified local path
payments_path = 'olist_order_payments_dataset.csv'  # Replace with the actual path to the file
order_payments = pd.read_csv(payments_path)

# Display basic information about the dataset
payments_info = order_payments.info()

# Analyze the distribution of payment methods
payment_methods_distribution = order_payments['payment_type'].value_counts(normalize=True)

# Calculate the average payment value per order
average_payment_value = order_payments['payment_value'].mean()


# Select only numerical columns for the correlation analysis
numerical_order_payments = order_payments.select_dtypes(include=['int64', 'float64'])

# Calculate summary statistics for numerical features
summary_statistics = numerical_order_payments.describe()

# Calculate the distribution of payment installments
installments_distribution = order_payments['payment_installments'].value_counts()

# Plot the histogram of payment values
plt.figure(figsize=(10, 6))
sns.histplot(order_payments['payment_value'], kde=True)
plt.title('Distribution of Payment Values')
plt.xlabel('Payment Value')
plt.ylabel('Frequency')
plt.show()

# Plot the boxplot to check for outliers in payment values
plt.figure(figsize=(10, 6))
sns.boxplot(x=order_payments['payment_value'])
plt.title('Boxplot of Payment Values')
plt.xlabel('Payment Value')
plt.show()

# Calculate the correlation matrix for numerical features
correlation_matrix = numerical_order_payments.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Payment Dataset')
plt.show()

failed_payment_distribution = order_payments['failed_total_payment'].value_counts(normalize=True)

# Plot the distribution of the 'failed_total_payment'
plt.figure(figsize=(8, 5))
sns.countplot(x=order_payments['failed_total_payment'])
plt.title('Distribution of Failed Total Payment')
plt.xlabel('Failed Total Payment')
plt.ylabel('Count')
plt.show()

# Analysis of payment value by failed payment status
plt.figure(figsize=(10, 6))
sns.boxplot(x='failed_total_payment', y='payment_value', data=order_payments)
plt.title('Payment Value by Failed Payment Status')
plt.xlabel('Failed Total Payment')
plt.ylabel('Payment Value')
plt.show()

# Print the distribution of failed payments
print("Distribution of Failed Total Payment:\n", failed_payment_distribution)

# Print the results
print("Summary Statistics:\n", summary_statistics)
print("\nInstallments Distribution:\n", installments_distribution)
print("\nCorrelation Matrix:\n", correlation_matrix)


'''Predict the payment method by using all other attributes as features'''

'''Predict in how many installments the user will finally pay the product based on payment method etc'''

payments_path = 'olist_order_payments_dataset.csv'  # Replace with the actual path to the file
order_payments = pd.read_csv(payments_path)

#remove the user_id

order_payments = order_payments.drop(['order_id', 'payment_sequential'], axis=1)

# Initialize the PaymentMethodPredictor with the DataFrame and target variable
predictor = PaymentMethodPredictor(order_payments[:50000], 'failed_total_payment')

# Preprocess the data
predictor.preprocess_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = predictor.split_data()

# Train the model
predictor.train_model(X_train, y_train)

# Evaluate the model
accuracy = predictor.evaluate_model(X_test, y_test)
print(f"Final Model accuracy: {accuracy}")

nn_predictor = PaymentMethodNNPredictor(order_payments[:50000], 'failed_total_payment')

# Preprocess the data
nn_predictor.preprocess_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = nn_predictor.split_data()

# Build the neural network model
nn_predictor.build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

# Train the model
nn_predictor.train_model(X_train, y_train)

# Evaluate the model
nn_predictor.evaluate_model(X_test, y_test)

# To predict the likelihood of failed payments
likelihoods = nn_predictor.model.predict(X_test)[:, 1]