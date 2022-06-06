#Sex and age classification using EEG dataset
#The EEG dataset was taken from OpenNeuro - https://openneuro.org/datasets/ds003969/versions/1.0.0

#First, we will perform explorarion of the dataset:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder



Participants = pd.read_csv('participants.tsv', sep='\t')

Participants.info()
#According to to info there is 1 null value in age of one of the participants so we will exclude this participant:
Participants_Age_no_na = Participants.loc[:,['age','gender']].dropna()

#Gender variable is given as m(male) or f(female) thus we will change to one-hot vector before creating histograms: 1-male, 0-female
label_encoder = LabelEncoder()
Gender_catagorical = label_encoder.fit_transform(Participants_Age_no_na.loc[:,'gender'])
print(Gender_catagorical)

#Distribution of age and gender:
Participants_Age_no_na['gender_catagorical'] = Gender_catagorical.tolist()
Participants_Age = Participants_Age_no_na['age']
Participants_Age.hist(bins=10)
plt.title('Age distribution histogram')
plt.xlabel('Age [$years$]')
plt.ylabel('Count')
plt.show()

#Show the percentages of females and males using pie chart:
Males = np.count_nonzero(Gender_catagorical)
Females = len(Participants_Age_no_na)-Males
plt.pie([Males,Females], labels=['Males','Females'], colors = ['steelblue', 'salmon'], autopct='%1.1f%%')
plt.title('Gender distribution')
plt.show()

#After all the matrices were saved perform train-test split, X should be the data (the matrices) and Y should be the labels
X = np.load('/Users/carme/Desktop/ML for physiological time series/Project/EEG/sub-001_eeg_sub-001_task-med1breath_eeg.bdf.npy')
