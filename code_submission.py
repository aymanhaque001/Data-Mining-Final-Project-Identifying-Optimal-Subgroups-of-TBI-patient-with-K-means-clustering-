# For reading csv file and creating dataframes
import pandas as pd

# For numerical computations
import numpy as np

# For preprocessing data
from sklearn import preprocessing

# For clustering
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

# For dimensionality reduction
from sklearn.decomposition import PCA

# For plotting graphs 
import matplotlib.pyplot as plt

# For dimensionality reduction
from sklearn.manifold import Isomap,TSNE

# For calculating internal validity metrics
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score,silhouette_score

import weka.core.jvm as jvm
import glob
import sys
import os
import weka.core.packages as pkg
from weka.core.converters import Loader
# This is for loading csv files
import weka.core.converters as converters
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel
import weka.core.packages as packages


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# How to split the dataset
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#from sklearn import cross_validation

# Preprocessing data
from sklearn.preprocessing import StandardScaler

# Evaluation metrics/scores
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


jvm.start(packages=True)


# root folder
file_gdrive_loc = './dataset/'

# file containing military information
military_file = file_gdrive_loc + 'tbi_military.csv'

# file containing age information 
age_file = file_gdrive_loc + 'tbi_age.csv'

# file containing year information
year_file = file_gdrive_loc + 'tbi_year.csv'

# read files into dataframes
military_data = pd.read_csv(military_file)
age_data = pd.read_csv(age_file)
year_data = pd.read_csv(year_file)

# remove all rows for which "age_group" reads "Total"
age_data= age_data.drop(age_data[age_data["age_group"]=='Total'].index)

# Encode age groups with numbers (ordinal) 
age_data_enc= age_data.replace(['0-17', '0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64','65-74', '75+'], [0,1,2,3,4,5,6,7,8,9])

# Encode the type of visit with numbers (ordinal) using severity with death being the most severe followed by hospitalization and then emergency department
age_data_enc= age_data_enc.replace(['Emergency Department Visit', 'Hospitalizations', 'Deaths'],[0,1,2])

# One hot encoding for column: "Injury Mechanism"
injury_mechanism_encoded=pd.get_dummies(age_data.injury_mechanism)
age_data_enc = pd.concat([age_data_enc, injury_mechanism_encoded], axis=1, sort= False)
del age_data_enc['injury_mechanism']

#Dropping rows with NaN values**
age_data_enc.dropna(inplace = True)
age_data_enc.sort_index()


# "Orignal data" is the data in the encoded form but not normalized form
original_data = age_data_enc

# Drop the columns "number_est" and "rate_est" since we do not use them in further analysis
original_data = original_data.drop(["number_est","rate_est"],axis=1)
#['age_group', 'type', 'Assault', 'Intentional self-harm','Motor Vehicle Crashes', 'Other or no mechanism specified','Other unintentional injury, mechanism unspecified','Unintentional Falls','Unintentionally struck by or against an object']


# Normalize the data
x = age_data_enc.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
age_data_enc_scaled = pd.DataFrame(x_scaled)
age_data_enc_scaled.columns = age_data_enc.columns
data = age_data_enc_scaled


def run_kmeans(k):
    '''
    This function runs kmeans clustering algorithm on the data using the provided number of k as input
    '''
    kmeans = KMeans(n_clusters=k).fit(data)
    return list(kmeans.labels_)


def run_tsne_and_visualize(data,result):
    '''
    This function takes a dataset and clustering results as input and uses TSNE dimensionality reduction to visualize cluster assignment
    '''
    x = StandardScaler().fit_transform(data)
    tsne = TSNE(n_components=2)
    Components = tsne.fit_transform(x) 
    componentDf = pd.DataFrame(data = Components, columns = ['Component1', 'Component2'])
    xvals = componentDf["Component1"]
    yvals = componentDf["Component2"]
    plt.scatter(xvals,yvals,c=result)
    plt.title("TSNE Visualization with k={}".format(len(list(set(list(result))))))
    plt.show()


def run_pca_and_visualize(data,result):
    '''
    This function takes a dataset and clustering results as input and uses PCA dimensionality reduction to visualize cluster assignment
    '''    
    x = StandardScaler().fit_transform(data)
    tsne = PCA(n_components=2)
    Components = tsne.fit_transform(x) 
    componentDf = pd.DataFrame(data = Components, columns = ['Component1', 'Component2'])
    xvals = componentDf["Component1"]
    yvals = componentDf["Component2"]
    plt.scatter(xvals,yvals,c=result)
    plt.title("PCA Visualization with k={}".format(len(list(set(list(result))))))
    plt.show()


# DB(min best)  - minimum score is zero, with lower values indicating better clustering
# SIl(max best) - The best value is 1 and the worst value is -1
# CH(max best)  - It is also known as the Variance Ratio Criterion.


# Kmeans for k = 2
k2_result = run_kmeans(2)
run_tsne_and_visualize(original_data,k2_result)
run_pca_and_visualize(original_data,k2_result)

ch_2 = calinski_harabasz_score(age_data_enc_scaled,k2_result)
db_2 = davies_bouldin_score(age_data_enc_scaled,k2_result)
si_2 = silhouette_score(age_data_enc_scaled,k2_result)



# Kmeans for k = 3
k3_result = run_kmeans(3)
# run_tsne_and_visualize(original_data,k3_result)
# run_pca_and_visualize(original_data,k3_result)
ch_3 = calinski_harabasz_score(age_data_enc_scaled,k3_result)
db_3 = davies_bouldin_score(age_data_enc_scaled,k3_result)
si_3 = silhouette_score(age_data_enc_scaled,k3_result)


# Kmeans for k = 4
k4_result = run_kmeans(4)
# run_tsne_and_visualize(original_data,k4_result)
# run_pca_and_visualize(original_data,k4_result)
ch_4 = calinski_harabasz_score(age_data_enc_scaled,k4_result)
db_4 = davies_bouldin_score(age_data_enc_scaled,k4_result)
si_4 = silhouette_score(age_data_enc_scaled,k4_result)
weighted_score_4 = (ch_4 + si_4 + (-1*db_4))/3

# Kmeans for k = 5
k5_result = run_kmeans(5)
result_df = pd.DataFrame()
result_df["Cluster results"] = k5_result
result_df.to_csv("k5.csv",index=False)

ch_5 = calinski_harabasz_score(age_data_enc_scaled,k5_result)
db_5 = davies_bouldin_score(age_data_enc_scaled,k5_result)
si_5 = silhouette_score(age_data_enc_scaled,k5_result)



print ("k2 score ",db_2)
print ("k3 score ",db_3)
print ("k4 score ",db_4)
print ("k5 score ",db_5)



# create a dictionary to map the clustering results and corresponding score
result_mapping = {
    'k2_result':weighted_score_2,
    'k3_result':weighted_score_3,
    'k4_result':weighted_score_4,
    'k5_result':weighted_score_5
}

name_mapping = {
    'k2_result':k2_result,
    'k3_result':k3_result,
    'k4_result':k4_result,
    'k5_result':k5_result
}


# ======== get the result with the highest weighted score ======================
selected_result = max(result_mapping, key=result_mapping.get) 
selected_result = name_mapping[selected_result]
selected_data = original_data
selected_data["cluster result"] = selected_result
selected_data_for_feature_selecion = selected_data
# ==============================================================================


# ============================Feature Selection ================================
# Create a dataframe 
dataframe = selected_data_for_feature_selecion
dataframe.to_csv("data_for_feature_selection.csv",index=False) 
# This is the list of dataframe columns
dataframe_columns = dataframe.columns.tolist()
# get the data csv file 
# covert data to weka format 
data = converters.load_any_file("data_for_feature_selection.csv")
# assign class to be last column 
data.class_is_last()

# Evolutionary Search 
search = ASSearch(classname="weka.attributeSelection.EvolutionarySearch",options=["-population-size","20","-generations", "20", "-crossover-probability","0.6"])

evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P","1","E","1"])
attsel = AttributeSelection()
attsel.folds(10)
attsel.crossvalidation(True)
attsel.seed(1)
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)
evl= Evaluation(data)
# number of attributes selected
quantity = attsel.number_attributes_selected
# positions of attributes in the columns list
feature_positions = attsel.selected_attributes[0:quantity]
chosen_columns = [dataframe_columns[position] for position in feature_positions]
# Add the label to the chosen columns.So that It will be at the end of the dataframe 
chosen_columns.extend(dataframe_columns[-1:])
e_df = pd.DataFrame()
e_df["evolutionarySearch"] = chosen_columns[0:-1]


final_df = pd.concat([e_df],ignore_index=True,axis=1)

headings = {
    0:"evolutionarySearch",
}
final_df=final_df.rename(columns=headings)
final_df.to_csv("dacosta final dataframe.csv")
print(final_df)
jvm.stop()


#--------------------Selected features-----------------------------

data_df = age_data_enc_scaled[['Assault','Intentional self-harm','Other or no mechanism specified','Other unintentional injury, mechanism unspecified','Unintentionally struck by or against an object']].values
label_df = selected_result

#------------------------------------------------------------------


MLP_df = pd.DataFrame()
header_name = "MLP"
MLP_df[header_name] = ["mean accuracy"] 
X = data_df
Y = np.array(label_df) 
learning_rates = [0.2,0.3]
momentums      = [0.1,0.2]
for lr in learning_rates:
    for m in momentums:
        kf = KFold(n_splits=10)
        clf = MLPClassifier(learning_rate_init = lr , momentum = m )
        accuracy = [] 
        for train_index,test_index in kf.split(X):
            X_train,X_test = X[train_index],X[test_index]
            Y_train,Y_test = Y[train_index],Y[test_index]                       
            result = clf.fit(X_train,Y_train)
            predictions = clf.predict(X_test)
            acc = accuracy_score(Y_test,predictions)
            accuracy.append(acc) 
        mean_accuracy = ((sum(accuracy))/float((len(accuracy))))*100
        MLP_df["LearningRate={} Momentum={}".format(lr,m)] = [mean_accuracy]
print(MLP_df)
MLP_df.to_csv("MLP_df.csv",index=False)











