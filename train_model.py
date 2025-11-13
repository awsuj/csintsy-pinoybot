import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
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


print("Starting model training process...")
data = 'final_annotations.csv'
try:
    df = pd.read_csv(data)
except FileNotFoundError:
    print(f"Error: {data} not found.")
    print("Please make sure the file is in the same directory.")
    exit()

df_clean = df[['word', 'label']].dropna() # remove nulls (NOT SURE IF NEEDED)
X_raw = df_clean['word']

print("Mapping labels to FIL, ENG, OTH...")
y = df_clean['label'].apply(map_labels)  # labeling labels to FIL, ENG, or OTH (NOT SURE IF NEEDED)

print(f"Loaded and cleaned {len(df_clean)} data points.")

print("New label distribution:")
print(y.value_counts())

# extract features
print("Extracting features...")
X_features = extract_features(X_raw.tolist())
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

print("\n--- 1. Tuning Model with Validation Set (using Macro F1-score) ---")

# 1. Define the settings (hyperparameters) you want to try
possible_depths = [10, 12, 16, 18, 20, 22, 24, 26, 28, 30]
best_depth = None
best_val_score = 0.0  # Keep track of the best score

# 2. Loop through each setting
for depth in possible_depths:
    print(f"Testing max_depth = {depth}...")

    # 3. Create and train a *temporary* model
    model_to_tune = DecisionTreeClassifier(random_state=42, max_depth=depth)
    model_to_tune.fit(X_train, y_train)

    # 4. Evaluate it on the VALIDATION set (the "practice exam")
    y_val_pred = model_to_tune.predict(X_val)
    # --- THIS IS THE KEY CHANGE ---
    # We use f1_score with average='macro' to handle imbalance
    val_score = f1_score(y_val, y_val_pred, average='macro')
    print(f"  Validation Macro F1: {val_score:.4f}")
    # ----------------------------

    # 5. Check if this is the best one so far
    if val_score > best_val_score:
        best_val_score = val_score
        best_depth = depth

print("\n--- 2. Tuning Complete ---")
print(f"Best max_depth found: {best_depth} (with {best_val_score:.4f} Macro F1)")

# 6. Now, train your FINAL model on X_train using the best setting
print("\n--- 3. Training Final Model ---")
final_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
final_model.fit(X_train, y_train)
print("Final model training complete.")

# 7. Generate a SIMPLE, READABLE tree image (for your report)
print("Generating simple decision tree image...")
plt.figure(figsize=(275, 50))  # A good size for a shallow tree
plot_tree(final_model,
          feature_names=X_features.columns.tolist(),
          class_names=final_model.classes_,
          filled=True,
          rounded=True,
          fontsize=5,)
plt.savefig('decision_tree_depth_20_1.png', dpi=300)
print("Saved simple tree image to 'decision_tree_simple.png'")

# 8. Evaluate the FINAL model on the TEST set (the "final exam")
print("\n--- 4. Final Evaluation on Test Set ---")
y_pred = final_model.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)  # Added digits=4 for more detail
print(report)

print("\n--- Feature Importance Report ---")

# Get feature names from your DataFrame
feature_names = X_features.columns.tolist()
# Get importances from the trained model
importances = final_model.feature_importances_

# Create a DataFrame to see them clearly
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

# Sort by importance, from 0.0 (unused) upwards
importance_df = importance_df.sort_values(by='importance', ascending=True)

print("--- UNUSED/LEAST IMPORTANT FEATURES ---")
print(importance_df.head(10))  # Shows the 10 least important

print("\n--- MOST IMPORTANT FEATURES ---")
print(importance_df.tail(10).sort_values(by='importance', ascending=False)) # Shows the 10 most important

# 9. Save the final, tuned model
model_filename = 'pinoybot_model_f1_validated_depth_20_1.pkl'
print(f"\n--- 5. Saving Final Model ---")
print(f"Saving trained model to {model_filename}...")
with open(model_filename, 'wb') as f:
    pickle.dump(final_model, f)

print("Process finished.")