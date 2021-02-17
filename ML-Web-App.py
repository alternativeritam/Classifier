import streamlit as st
from sklearn import datasets
from sklearn.neighbors import  KNeighborsClassifier
from sklearn .svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import  PCA
import matplotlib.pyplot as plt

st.title("ML models")

st.write("""
# diffrerent classifier

best one of them 

""")

dataset_name = st.sidebar.selectbox("Select DataBase", ("Iris Dataset","Breast Cancer","Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest"))


def get_dataset(dataset_name):
    if dataset_name=="Iris Dataset":
        data = datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x= data.data
    y= data.target

    return  x,y

x,y = get_dataset(dataset_name)

#st.write(y[0])
#st.write(x[0][0])

def add_parameter_ui(classifier_name):
    param = dict()
    if classifier_name=="KNN":
        k=st.sidebar.slider("K",1,15)
        param["K"] = k

    elif classifier_name=="SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        param["C"] = C
    else:
        max_depth = st.sidebar.slider("Max depth",2,16)
        estimator = st.sidebar.slider("estimator",1,100)
        param["max_depth"] = max_depth
        param["estimator"] = estimator
    return  param

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if classifier_name=="KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif classifier_name=="SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["estimator"],max_depth=params["max_depth"],random_state=42)
    return  clf

clf = get_classifier(classifier_name,params)

x_train , x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2,random_state=42)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")


# PLOTTING
import warnings
warnings. filterwarnings('ignore')
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2 , c=y,alpha=0.8, cmap="viridis")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

st.pyplot(fig)
