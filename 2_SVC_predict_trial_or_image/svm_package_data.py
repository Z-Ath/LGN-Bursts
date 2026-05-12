#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May  8 13:41:59 2026

Preprocessing used to package the data into spike counts for both Tonic and Burst firing during a
a stimulus period (0-0.25 ms following image) and processing period (0.35 to 0.7 ms following image). 

Creates labels on the data depending on either Image Classification (label per image shows 1-8) or 
on behavior (Hit Go vs Missed Go).

Data is lastly organized by if the image presented at the given time is either a familiar (repeat) image or
a new deviant (change) image. 

Counts are then stored as a MxN matrix where each row (M) is another image presentation (either familiar
or deviant) and each column (N) is the summed spike counts for the given neuron at said presentation.

Tonic and Burst were seperating by thresholding the ISI intervals and time between bursts (see tonic_burst)
 
Loads in data (using chosen session) from the Allensdk (downloaded using 
                                 from allensdk.brain_observatory.behavior.behavior_project_cache.\
                                 behavior_neuropixels_project_cache \
                                 import VisualBehaviorNeuropixelsProjectCache
                                 
Data was pre-packed into .pkl files with trial and session info along with raw spiketimes

Sessions used in analysis for paper:
    [1092466205, 1091039376, 1130349290, 1064639378, 1053925378,
     1096935816, 1122903357, 1108531612, 1104297538, 1081431006,
     1048196054, 1086410738, 1120251466, 1121607504, 1067781390,
     1065908084, 1115356973, 1112515874, 1108528422, 1063010385]

Outputs data into a structure with the following branch system:
    Response Type (ImageClass or BehaviorClass)
        -> Session Number['s' + number]
            -> Time period of interest (Stim or Proc)
                -> Spiking Type and ImageType (For Imageclass: TonicFaimilar,BurstFamiliar,
                                                               TonicDeviant,BurstDeviant,
                                               For BehaviorClass: Tonic, Burst)
            -> Eventlabels (EventlabelFamiliar, EventlabelDeviant)
            -> Class label where the image comes from (ClassSet) either A or B
    Sampling Rate (fs)
    Stim period time start in seconds (tstim_start)
    Stim period time end in seconds (tstim_end)
    Processing period time start in seconds (tproc_start)
    Processing period time end in seconds (tproc_end)
            
    
@author: loganbecker
"""

import numpy as np
from research_utils.spiking import tonic_burst
from funcs import pack_counts, mean_count_region
import pandas as pd
from joblib import load


#%% Load in data
fs = 1250; dt = 1/fs 
np.random.seed(6895)

datapath = '' # PUT PATH TO DAT HERE 
sess_list = np.load('sessions_list.npy') # Load in list of sessions used in analysis

# Images in each class (given by Allensdk)
classA = ['im005', 'im024', 'im034', 'im083', 'im087', 'im104', 'im111','im114']
classB = ['im012', 'im036', 'im044', 'im047', 'im078', 'im083', 'im111','im115']

# Time periods of interest
tstim_start, tstim_end = 0, 0.25 # Stim (in seconds)
tproc_start, tproc_end = 0.35, 0.7 # Proc (in seconds)

# Eventlabels
eventlabel_familiar,  eventlabel_deviant = [], [] # Event labels
eventlabel_img, eventlabel_img_grouped, eventlabel_grouped = [], [], []

# Store info for Image Classification
counts_familiarT_s, counts_deviantT_s, counts_familiarB_s, counts_deviantB_s = [], [], [], [] # Stim info
counts_familiarT_p, counts_deviantT_p, counts_familiarB_p, counts_deviantB_p = [], [], [], [] # Proc info

# Store info for Behavior Classification
counts_behaveT_p, counts_behaveT_s, counts_behaveB_p, counts_behaveB_s = [], [], [], []
class_ = []

for nn, sess in enumerate(sess_list):
    print(f'Current Session: {sess}')
    data_filename = f'{datapath}/session_{sess}'; # File name for saving
    session_data = load(f'{data_filename}/session_data.pkl') # Load in data structure from prepacked struct

    spiketimes = session_data['spiketimes']['LGd'] # Spike times from LGd
    nT = len(spiketimes) # Number of Neurons
    t = session_data['tvec'] # Time vector

    session_start = t[0] # Session start time (nonzero)
    t = t-session_start # Adjust times so we start at t = 0
    spiketimes = [spiketimes[x]-session_start for x in range(nT)] # Adjust spike times with new start time
    
    # Load in data and stimulus info
    stim_info = session_data['stim']  
    imgstart = stim_info['start_time']-session_start # Time image [resentation starts [adjusted for start of trial]
    imgend = stim_info['end_time']-session_start # Time image presentation ends
    isdeviant = np.array([False if pd.isna(x) else bool(x) for x in stim_info['is_change']]) # Vector saying if image deviants in given trial
    isdeviant_sham = np.array([False if pd.isna(x) else bool(x) for x in stim_info['is_sham_change']]) # Vector saying if image deviants in given trial
    img_name = np.char.rstrip(stim_info['image_name'],'_r') #Image Name Vector
    img_ids = np.unique(img_name)[:-2] # Get Unique image names
    nimg = len(img_ids) # Number of unique images
    img_ids_clean = np.char.rstrip(img_ids, '_r')   # remove suffix "_r"
    
    # Keep track of the class being used (set of images)
    if sorted(classA) != sorted(img_ids_clean):
        class_.append('classA')
    else:
        class_.append('classB')
    
    # Get Event Info
    response_info = session_data['responses']
    session_info = session_data['session']
    hg = response_info['hit_go']-session_start # Hit Go
    mg = response_info['missed_go']-session_start # Missed Go

    # Trim info to just be within first experimental protocol
    stoptime = np.array(session_info['stop_time'])[-1]-session_start
    imgstart = imgstart[imgend <= stoptime]
    imgend = imgend[:len(imgstart)]
    isdeviant = isdeviant[:len(imgstart)]
    img_name = img_name[:len(imgstart)]
    isdeviant_sham = isdeviant_sham[:len(imgstart)]
    
    # Convertor of img number to id
    id_to_num = {img_id: i for i, img_id in enumerate(img_ids)} # Converts img_id to a number
    
    # Rasterize spiking
    tstim = t[t<=stoptime+1]; 
    y = np.zeros((nT,len(tstim)-1))
    for i, st in enumerate(spiketimes):
        # For each spike time, only keep ones that go from 0 to stoptime
        inds = np.searchsorted(tstim, st,side='right')-1
        inds = inds[(inds >= 0) & (inds < len(tstim)-1)]
        # Conver to binary given by time vector
        y[i] = np.bincount(inds,minlength = len(tstim)-1)
    y[y > 1] = 1 # Just in case any spiking intervals are too small, make sure its binary

    # --- Seperate spikes to Tonic and Burst ---
    tonic_spike_times, burst_spike_times, y_tonic, y_burst = tonic_burst(y,tstim)
    
    # Periods of interest to look at 
    tstart = -0.25; tend = 1.0 # 250 ms before stim to 1 second after   
    tstart_idx = np.searchsorted(t,0.25) # Bins for 250 ms
    tend_idx = np.searchsorted(t,tend) # Bins till end from img
    tvec = np.arange(tstart,tend-dt,dt) # Tvec
    tol = 0.1; # Tolerance for around hit go time

    # Binning data in smaller time windows (10 ms)
    bin_width = 0.01 # Bin used for psth (ms)
    bins = np.arange(tstart, tend + bin_width, bin_width); binlen = len(bins)-1 # Bin vector 

    # Sort data by Image Label and get Counts
    eventlabel_hg, eventlabel_cr, eventlabel_mg, eventlabel_fa = [], [], [], [] # Event labels
    counts_familiarT_, counts_deviantT_, eventlabel_familiar_, eventlabel_deviant_ = [], [], [], []
    counts_familiarB_, counts_deviantB_ = [], []
    counts_behaveT, counts_behaveB, counts_behaveAll, eventlabel_behave = [], [], [], []
    eventlabel_img_ = []
    for imgname in img_ids:
        
        # *** Response Type for Image ***
        
        # --- Familiar Images ---
        img_ex_idx = np.where(np.array(img_name) == imgname)[0] # Find img idx
        isdeviant_ex = isdeviant[img_ex_idx] # Find where deviant 
        imgstart_ex = imgstart[img_ex_idx] # Find img start times
        valid_idx = np.where(~(isdeviant_ex[2:] | isdeviant_ex[1:-1]))[0] + 2 # Only include times where there is no deviant (2 delay)
        img_append = id_to_num[imgname]
        
        # Index for familiar images
        idx_start = np.searchsorted(tstim,imgstart_ex[valid_idx]) - tstart_idx # Start time idx [tstart - 0.25 ms]
        idx_end = idx_start + tstart_idx + tend_idx # End time idx [tstart + 1 s]
        
        # Bursting and Tonic Counts - Packages data into the correct format
        counts_familiarB_, eventlabel_familiar_ = pack_counts(burst_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append,counts_familiarB_,eventlabel_familiar_) # BURST
        counts_familiarT_, _ = pack_counts(tonic_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append,counts_familiarT_,[]) # TONIC
       
        # --- Deviant Images ---
        valid_idx = np.where(isdeviant_ex)[0] # Find where deviant
        imgstart_ex = imgstart_ex[valid_idx] # Find img start
        mask = np.any(np.abs(imgstart_ex[:, None] - hg[None, :]) < tol, axis=1) # Only use HIT GO deviants
        imgstart_ex = imgstart_ex[mask] # Adjust start with HG
        valid_idx_mask = valid_idx[mask] # Adjust idx with start 
        
        # Index for deviantd images
        idx_start = np.searchsorted(tstim,imgstart_ex) - tstart_idx
        idx_end = idx_start + tstart_idx + tend_idx
        
        # Bursting and Tonic Counts - Packages data into the correct format
        counts_deviantB_, eventlabel_deviant_ = pack_counts(burst_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append,counts_deviantB_,eventlabel_deviant_) # BURST
        counts_deviantT_, _ = pack_counts(tonic_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append,counts_deviantT_,[]) # TONIC
        
        
        # *** Response Type for Behavior ***
        isdeviant_ex = isdeviant[img_ex_idx] # Get when it deviantd vs familiar
        imgstart_ex = imgstart[img_ex_idx] # Get img start times
        isdeviantSham_ex = isdeviant_sham[img_ex_idx]
        img_append2 = id_to_num[imgname]
 
        # HIT GO (deviant)
        valid_idx = np.where(isdeviant_ex)[0] # Get only where there is a deviant
        imgstart_hg = imgstart_ex[valid_idx] # Img start times
        mask = np.any(np.abs(imgstart_hg[:, None] - hg[None, :]) < tol, axis=1) # Only want to look at hitgo
        imgstart_hg = imgstart_hg[mask]
        valid_idx_mask = valid_idx[mask]
        
        idx_start = np.searchsorted(t,imgstart_hg) - tstart_idx
        idx_end = idx_start + tstart_idx + tend_idx
        
        ytemp = np.stack([y_burst[:,s:e] for s, e in zip(idx_start, idx_end)],axis = 1)
        counts_behaveB, _ = pack_counts(burst_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append2,counts_behaveB,[])
        counts_behaveT, _ = pack_counts(tonic_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append2,counts_behaveT,[])
        counts_behaveAll, _ = pack_counts(spiketimes,tstim,idx_start,idx_end,
                                               tstart,bins,img_append2,counts_behaveAll,[])
        eventlabel_behave.append(np.zeros(len(idx_start)))
    
        g = len(idx_start)
        
        # MISSED GO (deviant)
        valid_idx = np.where(isdeviant_ex)[0] # Get only where there is a deviant
        imgstart_mg = imgstart_ex[valid_idx] # Img start times
        mask = np.any(np.abs(imgstart_mg[:, None] - mg[None, :]) < tol, axis=1) # Only want to look at hitgo
        imgstart_mg = imgstart_mg[mask]
        valid_idx_mask = valid_idx[mask]
        
        idx_start = np.searchsorted(t,imgstart_mg) - tstart_idx
        idx_end = idx_start + tstart_idx + tend_idx
        
        counts_behaveB, _ = pack_counts(burst_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append2,counts_behaveB,[])
        counts_behaveT, _ = pack_counts(tonic_spike_times,tstim,idx_start,idx_end,
                                               tstart,bins,img_append2,counts_behaveT,[])
        counts_behaveAll, _ = pack_counts(spiketimes,tstim,idx_start,idx_end,
                                               tstart,bins,img_append2,counts_behaveAll,[])
        eventlabel_behave.append(np.ones(len(idx_start)))
        
        eventlabel_img_.append([img_append2*np.ones(len(idx_start)+g)])
        
    # Package and group by label (0s then 1s)
    counts_familiarB_, counts_deviantB_ = np.vstack(counts_familiarB_), np.vstack(counts_deviantB_) # BURSTS
    counts_familiarT_, counts_deviantT_ = np.vstack(counts_familiarT_), np.vstack(counts_deviantT_) # TONICS

    eventlabel_familiar_, eventlabel_deviant_ = np.array(eventlabel_familiar_), np.array(eventlabel_deviant_) # LABELS
    nfamiliar, ndeviant = len(eventlabel_familiar_), len(eventlabel_deviant_) # Number of familiars and deviants
    
    # Store all the info for each session
    
    # Tonic & Bursts Processing
    counts_familiarB_p.append(mean_count_region(nfamiliar,counts_familiarB_,bins,tproc_start,tproc_end)) # familiar
    counts_deviantB_p.append(mean_count_region(ndeviant,counts_deviantB_,bins,tproc_start,tproc_end)) # deviant
    counts_familiarT_p.append(mean_count_region(nfamiliar,counts_familiarT_,bins,tproc_start,tproc_end)) # familiar 
    counts_deviantT_p.append(mean_count_region(ndeviant,counts_deviantT_,bins,tproc_start,tproc_end)) # deviant
    
    # Tonic and Bursts Stimulus 
    counts_familiarB_s.append(mean_count_region(nfamiliar,counts_familiarB_,bins,tstim_start,tstim_end)) # familiar
    counts_deviantB_s.append(mean_count_region(ndeviant,counts_deviantB_,bins,tstim_start,tstim_end)) # deviant
    counts_familiarT_s.append(mean_count_region(nfamiliar,counts_familiarT_,bins,tstim_start,tstim_end)) # familiar 
    counts_deviantT_s.append(mean_count_region(ndeviant,counts_deviantT_,bins,tstim_start,tstim_end)) # deviant
    
    # Package labels
    eventlabel_familiar.append(eventlabel_familiar_)
    eventlabel_deviant.append(eventlabel_deviant_)

    # PACKAGE LABELS
    counts_behaveB = np.vstack(counts_behaveB); counts_behaveT = np.vstack(counts_behaveT); 
    eventlabel_behave = np.concatenate(eventlabel_behave)
    eventlabel_img = np.hstack(eventlabel_img_); 
    
    unbeh, unbeh_ct = np.unique(eventlabel_behave,return_counts=True)
    print(unbeh,unbeh_ct)
    
    # Package and group by label (0s then 1s)
    nevent = len(eventlabel_behave)
    order = np.argsort(eventlabel_behave)
    counts_groupedB = counts_behaveB[order]
    counts_groupedT = counts_behaveT[order]
    
    eventlabel_grouped.append(eventlabel_behave[order])
    eventlabel_img_grouped.append(eventlabel_img[0,order])

    # Convert to means
    counts_behaveT_p.append(mean_count_region(nevent,counts_groupedT,bins,tproc_start,tproc_end))
    counts_behaveB_p.append(mean_count_region(nevent,counts_groupedB,bins,tproc_start,tproc_end))
    counts_behaveT_s.append(mean_count_region(nevent,counts_groupedT,bins,tstim_start,tstim_end))
    counts_behaveB_s.append(mean_count_region(nevent,counts_groupedB,bins,tstim_start,tstim_end))
    
#%% Wrap it all up
data = {}
data['ImageClass'] = {}
data['ImageClass'] = {'s'+str(k): {} for k in sess_list}

data['BehaviorClass'] = {}
data['BehaviorClass'] = {'s'+str(k): {} for k in sess_list}

for nn, sess in enumerate(sess_list):
    data['ImageClass']['s'+str(sess)]['Stim'] = {}
    
    data['ImageClass']['s'+str(sess)]['Stim']['TonicFamiliar'] = counts_familiarT_s[nn]
    data['ImageClass']['s'+str(sess)]['Stim']['BurstFamiliar'] = counts_familiarB_s[nn]
    data['ImageClass']['s'+str(sess)]['Stim']['TonicDeviant'] = counts_deviantT_s[nn]
    data['ImageClass']['s'+str(sess)]['Stim']['BurstDeviant'] = counts_deviantB_s[nn]
    
    data['ImageClass']['s'+str(sess)]['Proc'] = {}
    
    data['ImageClass']['s'+str(sess)]['Proc']['TonicFamiliar'] = counts_familiarT_p[nn]
    data['ImageClass']['s'+str(sess)]['Proc']['BurstFamiliar'] = counts_familiarB_p[nn]
    data['ImageClass']['s'+str(sess)]['Proc']['TonicDeviant'] = counts_deviantT_p[nn]
    data['ImageClass']['s'+str(sess)]['Proc']['BurstDeviant'] = counts_deviantB_p[nn]
        
    data['ImageClass']['s'+str(sess)]['EventlabelFamiliar'] = eventlabel_familiar[nn]
    data['ImageClass']['s'+str(sess)]['EventlabelDeviant'] = eventlabel_deviant[nn]
    data['ImageClass']['s'+str(sess)]['ClassSet'] = class_[nn]
    
    data['BehaviorClass']['s'+str(sess)]['Stim'] = {}
    data['BehaviorClass']['s'+str(sess)]['Stim']['Tonic'] = counts_behaveT_s[nn]
    data['BehaviorClass']['s'+str(sess)]['Stim']['Burst'] = counts_behaveB_s[nn]
    
    data['BehaviorClass']['s'+str(sess)]['Proc'] = {}
    data['BehaviorClass']['s'+str(sess)]['Proc']['Tonic'] = counts_behaveT_p[nn]
    data['BehaviorClass']['s'+str(sess)]['Proc']['Burst'] = counts_behaveB_p[nn]

    data['BehaviorClass']['s'+str(sess)]['Eventlabel'] = eventlabel_grouped[nn]
    data['BehaviorClass']['s'+str(sess)]['ImgLabel'] = eventlabel_img_grouped[nn]
    data['BehaviorClass']['s'+str(sess)]['ClassSet'] = class_[nn]

data['Classes'] = {}
data['Classes']['ClassA'] = classA
data['Classes']['ClassB'] = classB

data['Fs'] = fs
data['StimStart'] = tstim_start
data['StimEnd'] = tstim_end
data['ProcStat'] = tproc_start
data['ProcEnd'] = tproc_end
