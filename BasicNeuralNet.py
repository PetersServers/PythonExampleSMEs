from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pandas as pd

class PaymentMethodNNPredictor:
    def __init__(self, dataframe, target_variable):
        self.dataframe = dataframe
        self.target_variable = target_variable
        self.encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()
        self.features = None
        self.labels = None
        self.model = None

    def preprocess_data(self):
        # Encode categorical variables
        X = self.dataframe.select_dtypes(include=['object']).apply(self.encoder.fit_transform)
        y = self.dataframe[self.target_variable]

        # Encode the target variable
        self.labels = self.encoder.fit_transform(y)

        # Apply one-hot encoding to the features
        X_encoded = self.one_hot_encoder.fit_transform(X).toarray()
        self.features = pd.DataFrame(X_encoded, columns=self.one_hot_encoder.get_feature_names_out())

        # One-hot encode the labels for the neural network
        self.labels = to_categorical(self.labels)

    def split_data(self):
        # Split the data into training and testing sets
        return train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

    def build_model(self, input_dim, output_dim):
        # Define a simple neural network model
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        # Train the neural network model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        # Evaluate the neural network model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test loss: {loss}, Test accuracy: {accuracy}")
