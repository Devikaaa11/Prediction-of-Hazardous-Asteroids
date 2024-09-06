import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix,f1_score,accuracy_score,precision_score,recall_score
import pickle


df=pd.read_csv(r"D:\assignments dsa\project dsa\flask\dataset.csv")

df2=df.drop(columns=['id','spkid','full_name','pdes','name','prefix','equinox'])


#Reducing number of rows that gives pha_N to minimise the huge imalance within column,that might effect performance of the model.

df2['pha'] = df2['pha'].astype(str)
yes_df = df2[df2['pha'] == 'Y']
no_df = df2[df2['pha'] == 'N']
no_df['missing_values'] = no_df.isnull().sum(axis=1)
no_df_sorted = no_df.sort_values(by='missing_values', ascending=False)
num_to_remove = min(920000, len(no_df_sorted))
print(f"\n\nNumber of rows to remove: {num_to_remove}")
rows_to_remove = no_df_sorted.head(num_to_remove)
no_df_remaining = no_df_sorted.drop(rows_to_remove.index)
no_df_remaining = no_df_remaining.drop(columns=['missing_values'])
result_df = pd.concat([yes_df, no_df_remaining])
result_df.reset_index(drop=True, inplace=True)
print(f"\n\nResulting DataFrame Shape: {result_df.shape}")

df_test=result_df.copy()
df_test = df_test.dropna(subset=['diameter','diameter_sigma','albedo'])

result_df.drop(columns=['tp', 'per_y','moid_ld','epoch_mjd','epoch_cal'],inplace=True)

result_df.drop(columns=['sigma_ad','sigma_per','sigma_om','sigma_tp','sigma_ma','sigma_e','sigma_i','sigma_q'],inplace=True)


# Fit the model using H to predict diameter
reg = LinearRegression()
reg.fit(df_test[['H']], df_test['diameter'])

# Predict missing diameter values based on H
missing_diameter = result_df[result_df['diameter'].isnull()]
predicted_diameter = reg.predict(missing_diameter[['H']])
result_df.loc[result_df['diameter'].isnull(), 'diameter'] = predicted_diameter


result_df['diameter_sigma'].fillna(result_df['diameter_sigma'].median(), inplace=True)
result_df['albedo'].fillna(result_df['albedo'].median(), inplace=True)


retain_classes = ['APO', 'ATE', 'AMO', 'IEO']
result_df['class'] = result_df['class'].apply(lambda x: x if x in retain_classes else 'Other')


droped_orbit_df=result_df.copy()
droped_orbit_df=result_df.drop(['orbit_id'],axis='columns')

num_droped_orbit_df=droped_orbit_df.select_dtypes(include=['int64','float64'])
cat_droped_orbit_df=droped_orbit_df.select_dtypes(include=['object'])

#encoding
encoded_df = pd.get_dummies(cat_droped_orbit_df, columns=['class', 'neo'])
print(encoded_df.head())
boolean_columns = encoded_df.columns.drop('pha')
encoded_df[boolean_columns] = encoded_df[boolean_columns].astype(int)
print("\n\n",encoded_df.head())


le = LabelEncoder()
encoded_df['pha'] = le.fit_transform(encoded_df['pha'])
encoded_df = encoded_df.astype(int)
print(encoded_df.head())

combined_df = pd.concat([num_droped_orbit_df, encoded_df], axis=1)
combined_df.head()

#splitting
x = combined_df.drop('pha', axis=1)
y = combined_df['pha']
x.head()
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#handling imbalance
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(x_train, y_train)


#scaling
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)
x_test = scaler.transform(x_test)

#model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=10, random_state=42, n_estimators=100)
rf_model.fit(X_resampled, y_resampled)
y_pred_rf = rf_model.predict(x_test)

print("\n\nTest Accuracy\n\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nF1 Score:", f1_score(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))

#pickling model

with open('model.pkl','wb') as model_file:
    pickle.dump(rf_model,model_file)


with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler,scaler_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(le,encoder_file)


# Save the column names used in one-hot encoding
with open('one_hot_columns.pkl', 'wb') as f:
    pickle.dump(encoded_df.columns.tolist(), f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(x.columns.tolist(), f)


print("Model and preprocessing tools saved successfully!")
