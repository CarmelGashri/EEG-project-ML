from mne.io import read_raw_bdf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
import matplotlib.pyplot as plt
import numpy as np

# path='./sub-001_eeg_sub-001_task-med1breath_eeg.bdf'
path = '/Users/carme/Desktop/ML for physiological time series/Project/EEG/sub-001_eeg_sub-001_task-med1breath_eeg.bdf'
raw_eeg = read_raw_bdf(path,preload=True)

#for i in [0, 5, 50, 75]:
#    channel_i = raw_eeg[raw_eeg.ch_names[i]]
#    plt.plot(channel_i[1], np.squeeze(channel_i[0]))
#    plt.show()

channel_i = raw_eeg[raw_eeg.ch_names[38]]
plt.plot(channel_i[1], np.squeeze(channel_i[0]))
plt.show()

channels_to_filter = np.arange(0,72)

#As can be seen, the data is noisy, therefore we will first perform pre-processing.
#First step - band pass filtering of 1-45Hz (different from the paper which performed 4-45Hz filtering) since this is a meditation task therefore lower frequencies should be kept in the data
#We will use IIR filter (backward-foreward/filtfilt) so no phase response

Filtered_eeg = raw_eeg.filter(l_freq=1,h_freq=45,picks=channels_to_filter,filter_length='auto',method='iir')

#Filtered_eeg = raw_eeg.filter(l_freq=1,h_freq=45,filter_length='auto',method='iir')

channel_i_filt = Filtered_eeg[Filtered_eeg.ch_names[38]]
plt.plot(channel_i_filt[1], np.squeeze(channel_i_filt[0]))
plt.show()

plt.plot(channel_i[1], np.squeeze(channel_i[0]),label='raw')
plt.plot(channel_i_filt[1], np.squeeze(channel_i_filt[0]),label='Band pass filtered')
plt.xlim(0,1)
plt.legend(bbox_to_anchor=(1.124, 0.4))
plt.show()

#50Hz notch filter (India's power line frequency)
Filtered_eeg_notch = Filtered_eeg.notch_filter(np.arange(50, 251, 50),method='fir',picks=channels_to_filter,phase='zero-double')

#Filtered_eeg_notch = Filtered_eeg.notch_filter(np.arange(50, 251, 50),method='fir',phase='zero-double')

channel_i_notch = Filtered_eeg_notch[Filtered_eeg_notch.ch_names[38]]

plt.plot(channel_i_filt[1], np.squeeze(channel_i_filt[0]),label='Band pass filtered')
plt.plot(channel_i_notch[1], np.squeeze(channel_i_notch[0]),label='Band pass and notch filtered')
plt.xlim(0,10)
plt.legend(bbox_to_anchor=(1.124, 0.4))
plt.show()

#EOG + ECG artifacts removal
#First lets examine the extrenal channels:

for i in [64, 65, 66, 67, 68, 69, 70, 71]:
    channel_i_i = Filtered_eeg_notch[Filtered_eeg_notch.ch_names[i]]
    plt.plot(channel_i_i[1], np.squeeze(channel_i_i[0]))
    plt.xlim(0, 5)
    plt.show()

#As can be seen, electrode 70 recorded ECG data, according to the paper 6 electrodes were used to record EOG data
#Therefore, we will use electrode 70 for ECG artifact removal and electrodes 64-69 for EOG artifact removal

ica = ICA(n_components=15, max_iter='auto', random_state=15)
ica.fit(Filtered_eeg_notch)
ica
ica.plot_sources(Filtered_eeg_notch, show_scrollbars=False)

EOG_ch_names =  ["EXG1", "EXG2", "EXG3","EXG4","EXG5","EXG6"]
ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(Filtered_eeg_notch,ch_name=EOG_ch_names)
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)

# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(Filtered_eeg_notch, show_scrollbars=False)


ica.exclude = []
# find which ICs match the EOG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(Filtered_eeg_notch,ch_name='EXG7')
ica.exclude = ecg_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(ecg_scores)

# plot ICs applied to raw data, with ECG matches highlighted
ica.plot_sources(Filtered_eeg_notch, show_scrollbars=False)

Filtered_eeg_notch_ICA = ica.apply(Filtered_eeg_notch)
channel_i_i = Filtered_eeg_notch[Filtered_eeg_notch.ch_names[38]]
plt.plot(channel_i_i[1], np.squeeze(channel_i_i[0]))
plt.xlim(0, 5)
plt.show()