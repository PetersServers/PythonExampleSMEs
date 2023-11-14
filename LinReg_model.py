from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

class PaymentMethodPredictor:
    def __init__(self, dataframe, target_variable):
        self.dataframe = dataframe
        self.target_variable = target_variable
        self.model = LogisticRegression(max_iter=1000)
        self.encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()
        self.features = None
        self.labels = None

        print(self.dataframe)
    def preprocess_data(self):
        # Encode categorical variables
        print("preprocessing data")
        X = self.dataframe.select_dtypes(include=['object']).apply(self.encoder.fit_transform)
        y = self.dataframe[self.target_variable]

        # Encode the target variable
        self.labels = self.encoder.fit_transform(y)

        # Apply one-hot encoding to the features
        X_encoded = self.one_hot_encoder.fit_transform(X).toarray()
        self.features = pd.DataFrame(X_encoded, columns=self.one_hot_encoder.get_feature_names_out())

    def split_data(self):
        # Split the data into training and testing sets
        return train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)

        # Predict on the training data
        y_train_pred = self.model.predict(X_train)

        # Calculate and print the training accuracy
        training_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Training accuracy: {training_accuracy}")

    def evaluate_model(self, X_test, y_test):
        # Predict and evaluate the model
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

