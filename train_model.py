import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from feature_extractor import extract_features
from sklearn.metrics import f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

"""
Annotations are labeled as 'FIL', 'ENG', or 'OTH'
"""
def map_labels(label):
    label = str(label)

    if label in ['FIL', 'CS']: # if a word is labeled as CS, it is used in FIL context
        return 'FIL'

    if label == 'ENG':
        return 'ENG'

    return 'OTH' # NE, SYM, NUM, EXPR, ABB

# Read training data from the file
print("Starting model training process...")
data = 'final_annotations.csv'
try:
    df = pd.read_csv(data)
except FileNotFoundError:
    print(f"Error: {data} not found.")
    exit()

df_clean = df[['word', 'label']].dropna() # removes any nulls
raw_word = df_clean['word']

print("Mapping labels to FIL, ENG, OTH...")
y = df_clean['label'].apply(map_labels)  # categorizing labels as FIL, ENG, or OTH

print(f"Loaded and cleaned {len(df_clean)} data points.")
print("New label distribution:")
print(y.value_counts())

# extract features
word_features = extract_features(raw_word.tolist())
print("Extracting features...")
print(f"Features extracted. Shape: {word_features.shape}")

# List of columns that return strings
categorical_cols = [
    'f_get_language',
    'f_oth_filter',
    'f_has_pair_vowel_word_duplication',
    'f_prefix_fil',
    'f_infix_fil',
    'f_suffix_fil',
    'f_eng_bigrams',
    'f_get_suffix_eng'
]

print("Encoding categorical features...")

# 2. Make a copy to work on
word_features_encoded = word_features.copy()

# Convert categorical features into numbers, unknown val prevents errors
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit the encoder and transform the data
word_features_encoded[categorical_cols] = encoder.fit_transform(word_features[categorical_cols])
print("Encoding done.")

# split data
print("Splitting data (70-15-15)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    word_features_encoded, y,
    test_size = 0.30,
    random_state = 42,
    stratify = y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size = 0.50,
    random_state = 42,
    stratify = y_temp
)
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

print("\n--- Tuning Model with Validation Set (using Macro F1-score) ---")
possible_depths = [6,7,8,9,10,11]
best_depth = None
best_val_score = 0.0
for depth in possible_depths:
    print(f"Testing max_depth = {depth}...")

    model_to_tune = DecisionTreeClassifier(random_state=42, max_depth=depth)
    model_to_tune.fit(X_train, y_train)
    y_val_pred = model_to_tune.predict(X_val)

    val_score = f1_score(y_val, y_val_pred, average='macro')
    print(f"  Validation Macro F1: {val_score:.4f}")

    if val_score > best_val_score: #save the best score
        best_val_score = val_score
        best_depth = depth

print("\n--- Tuning Complete ---")
print(f"Best max_depth found: {best_depth} (with {best_val_score:.4f} Macro F1)")

print("\n--- Train Model ---")
final_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
final_model.fit(X_train, y_train)
print("Model training complete")

# Generate an image of our decision tree
print("Generating decision tree...")
plt.figure(figsize=(200, 50))
plot_tree(final_model,
          feature_names=word_features.columns.tolist(),
          class_names=final_model.classes_,
          filled=True,
          rounded=True,
          fontsize=6,)
plt.savefig('decision_tree.png', dpi=300)
print("Saved decision tree image to 'decision_tree.png'")

print("\n--- Final Evaluation on Test Set ---")
y_pred = final_model.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)  #Until 4 decimal pts
print(report)

print("\n--- Feature Importance Report ---")

feature_names = word_features.columns.tolist()
importances = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort from least to most important
importance_df = importance_df.sort_values(by='Importance', ascending=True)

print("\n   --- UNUSED/LEAST IMPORTANT FEATURES ---")
print(importance_df.head(10))  # Show the 10 least used features
print("\n   --- MOST IMPORTANT FEATURES ---")
print(importance_df.tail(10).sort_values(by='Importance', ascending=False)) # Show the 10 most used features

# Save model and encoder
model_filename = 'pinoybot_model_f1_validated_depth_11.pkl'
encoder_filename = 'pinoybot_encoder_depth_11.pkl'

print(f"\n--- Saving Model ---")
print(f"Saving trained model to {model_filename}...")
with open(model_filename, 'wb') as f:
    pickle.dump(final_model, f)

print(f"Saving encoder to {encoder_filename}...")
with open(encoder_filename, 'wb') as f:
    pickle.dump(encoder, f)
print("Process finished")