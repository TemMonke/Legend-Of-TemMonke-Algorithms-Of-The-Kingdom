import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'SMSSpamCollection.txt'

# Read the dataset into a pandas DataFrame
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        label, text = line.strip().split('\t', 1)
        data.append((label, text))

dataframe = pd.DataFrame(data, columns=['Label', 'Text'])

# Split data into training and prediction sets
train_data = dataframe.iloc[:2000]  # First 2000 lines for training
predict_data = dataframe.iloc[2000:5000].copy()  # Next 3000 lines for prediction

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['Text'])
y_train = train_data['Label']

# Train the SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Prepare the prediction data
X_predict = vectorizer.transform(predict_data['Text'])

# Predict spam or ham
predict_data['Predicted_Label'] = model.predict(X_predict)

# Save the updated file with predictions
updated_file_path = 'Updated_SMSSpamCollection_SVM.txt'
predict_data.to_csv(updated_file_path, sep='\t', index=False, header=False)

print(f"Updated file with predictions saved to: {updated_file_path}")
