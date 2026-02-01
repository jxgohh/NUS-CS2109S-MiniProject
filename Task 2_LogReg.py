import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
class ciphertext_logreg():
    def __init__(self):
        self.DATA = pd.DataFrame()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def load_data(self, csv):
        df= pd.read_csv(csv)
        self.DATA = df
        return df

    def split_data(self, df, test_size=0.2):
        X = df['text']
        y = df['class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, max_iter=1000):
        model = LogisticRegression(max_iter=max_iter, class_weight='balanced')
        model.fit(self.X_train, self.y_train)
        self.model = model
        return model

    def predict(self, ciphertexts, model):
        predictions = model.predict(ciphertexts)
        return predictions

    def extract_char_features(self, X_train, X_test):
        all_chars = set()
        for text in self.X_train:
            all_chars.update(text)
        
        null_char = '\0'
        all_chars.add(null_char)
        
        char_list = sorted(list(all_chars))
        char_to_idx = {ch: i for i, ch in enumerate(char_list)} # Seen characters maaping to indices
        
        max_len = max(len(text) for text in self.X_train) + 10 # Add padding of 10 char in case unseen data is longer

        def pad_text(text):
            return text + null_char * (max_len - len(text))
        
        self.X_train = [pad_text(text) for text in self.X_train]
        self.X_test = [pad_text(text) for text in self.X_test]

        def text_to_features(text):
            return np.array([char_to_idx[ch] for ch in text])

        self.X_train = np.array([text_to_features(text) for text in self.X_train])
        self.X_test = np.array([text_to_features(text) for text in self.X_test])
        
        return char_to_idx, max_len, self.X_train, self.X_test

    def evaluate(self, X_train, X_test, y_train, y_test, model):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print("Train Accuracy", train_acc)
        print("Test Accuracy", test_acc)
        return

m = ciphertext_logreg()
df = m.load_data('data/cipher_objective.csv')

X_train, X_test, y_train, y_test = m.split_data(df, test_size=0.2)

charset, max_len, X_train, X_test = m.extract_char_features(X_train, X_test)

model = m.train(max_iter=1000)
print(max_len)
print(charset)

m.evaluate(X_train, X_test, y_train, y_test, model)