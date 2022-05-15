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
xlbl = ['Age [$years$]','Gender [$N.U$]']
axarr = Participants_Age_no_na.hist(bins=10, figsize=(20, 15))
for idx, ax in enumerate(axarr.flatten()):
    ax.set_xlabel(xlbl[idx])
    ax.set_ylabel("Count")
plt.show()