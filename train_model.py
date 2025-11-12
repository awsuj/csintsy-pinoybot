import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from feature_extractor import extract_features

print("Starting model training process...")

# load dataset
try:
    df = pd.read_csv('final_annotations.csv')
except FileNotFoundError:
    print("Error: 'final_annotations.csv' not found.")
    print("Please make sure the file is in the same directory.")
    exit()

# data cleaning
df_clean = df[['word', 'label']].dropna()

X_raw = df_clean['word']
y = df_clean['label']

print(f"Loaded and cleaned {len(df_clean)} data points.")

# extract features
print("Extracting features...")
X_features = extract_features(X_raw)
print(f"Features extracted. Shape: {X_features.shape}")

# split data
print("Splitting data (70-15-15)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_features, y,
    test_size = 0.30,
    random_state = 42,
    stratify = y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size = 0.50, # 50% of 30% is 15%
    random_state = 42,
    stratify = y_temp
)
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# decision tree training
print("Training Decision Tree model...")
model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)
print("Model training complete.")

# model evaluation
print("\n--- Model Evaluation on Test Set ---")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# save model
model_filename = 'pinoybot_model.pkl'
print(f"Saving trained model to {model_filename}...")
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print("Process finished.")
