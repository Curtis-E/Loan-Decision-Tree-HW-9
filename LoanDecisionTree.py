import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

#https://www.kaggle.com/datasets/burak3ergun/loan-data-set
df = pd.read_csv("loan_data_set.csv")
df.dropna(inplace=True)
df.drop(columns=['Loan_ID'], inplace=True)

cat_columns = df.select_dtypes("object").columns

# Prints label
df[cat_columns] = df[cat_columns].astype("category")
cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
                             range(len(df[cat_columns[i]].cat.categories))} for i in range(len(cat_columns))}
print(cat_dict)

# Makes categorical features as integer codes
df[df.select_dtypes("category").columns] = df[df.select_dtypes("category").columns].apply(lambda x: x.cat.codes)
df.dropna(inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train a decision tree classifier with max depth of 3
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Prints train and testing accuracy
print(f"Train Score: {clf.score(X_train, y_train):.3f}")
print(f"Test Score: {clf.score(X_test, y_test):.3f}")

# Shows Confusion Matrix
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# Plots Decision Tree
plt.figure(figsize=(15, 15))
plot_tree(clf, filled=True, feature_names=df.columns[1:], class_names=['No', 'Yes'])
plt.show()
