import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
fn = r'C:\Users\DELL I5558\Desktop\Python\NSW-ER01.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:22].astype(float)
Y = dataset[:, 22]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=21)

model_lin = SVC(kernel='linear', C=1, gamma=10)
model_rbf = SVC(kernel='rbf', C=1, gamma=10)

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=21)

pipeline_lin = make_pipeline(scaler, model_lin)
pipeline_rbf = make_pipeline(scaler, model_rbf)

X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.fit_transform(X_test)
model_lin.fit(X_scaled_train, y_train)
model_rbf.fit(X_scaled_train, y_train)
labels_lin = model_lin.predict(X_scaled_test)
labels_rbf = model_rbf.predict(X_scaled_test)

model_lin = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=1000, random_state=0)
transformed_lin = model_lin.fit_transform(X_scaled_test)
xs = transformed_lin[:, 0]
ys = transformed_lin[:, 1]
plt.scatter(xs, ys, c=labels_lin)
plt.show()

model_rbf = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=1000, random_state=0)
transformed_rbf = model_rbf.fit_transform(X_scaled_test)
xq = transformed_rbf[:, 0]
yq = transformed_rbf[:, 1]
plt.scatter(xq, yq, c=labels_rbf)
plt.show()
