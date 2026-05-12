#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:10:50 2026

Data created and saved from MakeSpikeCountData.py

@author: loganbecker
"""

import numpy as np
from svm_decoding_funcs import (boxsig, build_lda, run_roc_per_image, run_roc_shuffle, run_roc_per_behavior)
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.stats as st
from tqdm import trange

#%% Load in data and set basic parameter
np.random.seed(6895)

sess_list = np.load('sessions_list.npy') # Load in list of sessions used in analysis
nsess = len(sess_list) # Number of sessions used (was 20 for paper)
allcounts_ = np.load('SpikeCountsData_allrate.npy',allow_pickle=True).item() # Load in pre-sorted spiking info
savefig = False

fs = 1250; dt = 1/fs  # Sampling rate
nimg = 8 # Number of unique images
nsub = 1000 # Number of permutations when picking training set

#%% Part 1) Decoding Behavior (Hit Go vs Missed Go)
behaveclass_data = allcounts_['BehaviorClass'] # Load in Behavior data set
counts_behaveT = [behaveclass_data[x]['Proc']['Tonic'] for x in behaveclass_data] # Tonic Activty
counts_behaveB = [behaveclass_data[x]['Proc']['Burst'] for x in behaveclass_data] # Burst Activity
eventlabels_behave = [behaveclass_data[x]['Eventlabel'] for x in behaveclass_data] # Behavior Labels (HG (0) vs MG (1))
imglabels_behave = [behaveclass_data[x]['ImgLabel'] for x in behaveclass_data] # Img Labels (0-8)
class_sets =  [behaveclass_data[x]['ClassSet'] for x in behaveclass_data] # Load classes (subset of images used)

# set SVM type to use (Linear results shown in paper)
svm_type = 'linear' # Option for nonlinear or logistic if needed

# Test for both tonic and burst case
keys_firing = ['Tonic','Burst'] 

# Set up results to save across all sessions used, saving ROC AUC information
roc_out = {k: np.zeros(nsess) for k in keys_firing} # ROC AUC
stdroc_out = {k: np.zeros(nsess) for k in keys_firing} # STD ROC AUC
rocShuff_out = {k: np.zeros(nsess) for k in keys_firing} # Shuffled ROC AUC
pred_out = [] # Store predicted labels from fits
nTs = np.zeros(nsess) # Store number of neurons per session
nbehave = np.zeros([nsess,2]) # Store number of behaviors per session (Hg and MG)

print(f'Behavior: Fitting with {svm_type} over {nsess} sessions')
for zz in trange(nsess):
    
    # --- Load in Data and build LDAs ---
    countsT_ = counts_behaveT[zz] # Tonic Counts
    countsB_ = counts_behaveB[zz] # Burst Counts
    eventlabels_ = eventlabels_behave[zz] # Behavior Labels
    imglabel_ = imglabels_behave[zz] # Img Labels
    
    nbehave[zz,0] = np.sum(eventlabels_==0) # Number of HGs
    nbehave[zz,1] = np.sum(eventlabels_==1) # Number of MGs

    nN = countsT_.shape[1] # Number of neurons (same for both tonic and burst)
    nTs[zz] = nN
    
    # Build LDA; One for tonic and one for burst. Results set up to get ROC info out
    lda_tonic_roc = build_lda(countsT_,eventlabels_,svm_type,metric_type='roc') # TONIC LDA
    lda_burst_roc = build_lda(countsB_,eventlabels_,svm_type,metric_type='roc') # BURST LDA
    
    # --- Compile accuracies over nsub familiars (to capture variablity with subsampling) ---
    roc_data_ = {k: np.zeros(nsub) for k in keys_firing}
    roc_data_shuff_ = {k: np.zeros(nsub) for k in keys_firing}
    roc_curves_ = {k: [] for k in keys_firing}
    preds = []
    for ns in range(nsub):
        
        # --- Seperate data into training and test sets ---
        # Select random images for training and test (unique set of images for each to eliminate image effect)
        selected_labels = np.random.choice(np.unique(imglabel_),nimg,replace=False) # Randomize labels
        train_labels = selected_labels[:nimg//2] # Set images used for training
        test_labels = selected_labels[nimg//2:] # Set images used for testing
        
        # Training Set
        countsT_train = countsT_[np.isin(imglabel_,train_labels)] # Training Tonic
        countsB_train = countsB_[np.isin(imglabel_,train_labels)] # Training Burst
        eventlabel_train = eventlabels_[np.isin(imglabel_,train_labels)] # Training Behavior Labels
        
        # Test Set
        countsT_test = countsT_[np.isin(imglabel_,test_labels)] # Training Tonic
        countsB_test = countsB_[np.isin(imglabel_,test_labels)] # Training Burst
        eventlabel_test = eventlabels_[np.isin(imglabel_,test_labels)] # Training Behavior Labels
        
        # --- Find minimum number of training events (to equalize training) ---
        idx1 = np.sum(eventlabel_train==0) # Number of HIT GOs
        min_num_train = np.min([idx1,len(eventlabel_train)-idx1]) # Min number is Number of HG or Number of MG
        
        # Equlize Training set across HG and MG
        countsB_train_sub = np.zeros((2*min_num_train,nN))
        countsT_train_sub = np.zeros((2*min_num_train,nN))
        for i in range(2): # 2 conidtions (HG and MG)
            idx = np.where(eventlabel_train == i)[0] # Find all HG 
            idx_sub = np.random.permutation(len(idx))[:min_num_train] # Random familiar events
            countsB_train_sub[i*min_num_train:(i+1)*min_num_train,:] = countsB_train[idx[idx_sub],:] # Subsample Burst
            countsT_train_sub[i*min_num_train:(i+1)*min_num_train,:] = countsT_train[idx[idx_sub],:] # Subsample Train
        eventlabel_train_sub = np.familiar(range(2),min_num_train); # Subsample labels
        
        # Baseline to use for 'chance' in PR ACU
        baseline_ = np.sum(eventlabel_test)/len(eventlabel_test) # Number of HGs as baseline for test (chance)

        # Store true and predicted labels
        y_pred_ = np.zeros(((len(countsB_test)),3))  # Store labels
        y_pred_[:,0] = eventlabel_test # True labels
        
        # --- FIT AND GET ROC INFO ---
        roc_data_['Tonic'][ns], _, curves_h_T, _ = run_roc_per_behavior(lda_tonic_roc,countsT_train_sub, countsT_test, 
                                                                        eventlabel_train_sub,eventlabel_test) # Tonic
        roc_curves_['Tonic'].append(curves_h_T) # Curves

        roc_data_['Burst'][ns], _, curves_h_B, _ = run_roc_per_behavior(lda_burst_roc, countsB_train_sub, countsB_test, 
                                                                        eventlabel_train_sub,eventlabel_test) # Bursts    
        # Shuffle controls
        roc_data_shuff_['Tonic'][ns] = run_roc_shuffle(lda_tonic_roc,countsT_train_sub, countsT_test,
                                                       eventlabel_train_sub,eventlabel_test, n_classes=2)
        roc_data_shuff_['Burst'][ns] = run_roc_shuffle(lda_burst_roc,countsB_train_sub, countsB_test, 
                                                       eventlabel_train_sub,eventlabel_test, n_classes=2)
        preds.append(y_pred_)


    pred_out.append(preds) # Get predictions for each sessions

    # Average curves across shuffles for this session for Tonic and Burst
    mean_fpr = np.linspace(0, 1, 100)
    for k in keys_firing:
        roc_out[k][zz] = np.mean(roc_data_[k]) # Average ROC for given spike type in a given session
        stdroc_out[k][zz] = np.std(roc_data_[k]) # STD ROC for given spike type in a given session
        rocShuff_out[k][zz] = np.mean(roc_data_shuff_[k]) # Average ROC Shuffled in a given session
        
#%% Plot ROC results
# --- TONIC | BURST ROC Box Plot ---
# Box plot parameters
_, p_ = st.wilcoxon(roc_out['Tonic'],roc_out['Burst'])
_, pTbase_ = st.wilcoxon(roc_out['Tonic'],rocShuff_out['Tonic'], alternative='greater')
_, pBbase_ = st.wilcoxon(roc_out['Burst'],rocShuff_out['Burst'], alternative='greater')

pos_ = [0,0.3,0.9,1.2] # Positions 
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Colors
cls_ = [pltcolors[0],'grey',pltcolors[1],'grey'] # Colors
lbs_ = ['Test familiar','Shuffle','Test deviant','Shuffle'] # Labels

roc_together = [roc_out['Tonic'],rocShuff_out['Tonic'],
                roc_out['Burst'],rocShuff_out['Burst']] # Pack data together
rs = np.random.randn(4,nsess)*0.02 # Jitter for xlabels

figbox = plt.figure(1); plt.clf()
# Plot box plots
plt.boxplot(roc_together,positions=pos_,widths=0.3,showfliers=False,whis=[0, 100]) # Box plots
for i in range(len(pos_)):
    plt.scatter(pos_[i]*np.ones(nsess)+rs[i],roc_together[i],
                color=cls_[i],label=lbs_[i]) # Tonic Scatter
plt.ylabel('ROC AUC')
ax = plt.gca()
# Plot significances
boxsig(p_, pos_[0], pos_[2], 1.1, 0.05, ax)
boxsig(pTbase_, pos_[0], pos_[1], 1, 0.05, ax)
boxsig(pBbase_, pos_[2], pos_[3], 1, 0.05, ax)

# Add connecting lines
for i in range(nsess):
    plt.plot([rs[0][i],pos_[2]+rs[2][i]],[roc_together[0][i],roc_together[2][i]],'grey',alpha=0.3)

chance = 0.5 
width = 0.3

plt.ylim([0,1.5]);
plt.hlines(chance,pos_[0]-width,pos_[-1]+width,'k',linestyles='--')
plt.xticks([0.15,1.05],['Tonic','Burst']);
plt.xlim([pos_[0]-width,pos_[-1]+width])
plt.tight_layout()

#%% Image decoding
imgclass_data = allcounts_['ImageClass'] # Load in Image data set
counts_familiarT = [imgclass_data[x]['Proc']['TonicFamiliar'] for x in imgclass_data] # Familiar Tonic Activty
counts_deviantT = [imgclass_data[x]['Proc']['TonicDeviant'] for x in imgclass_data] # Deviant Tonic Activity
counts_familiarB = [imgclass_data[x]['Proc']['BurstFamiliar'] for x in imgclass_data] # Familiar Burst Activty
counts_deviantB = [imgclass_data[x]['Proc']['BurstDeviant'] for x in imgclass_data] # Deviant Burst Activity
eventlabel_familiar = [imgclass_data[x]['EventlabelFamiliar'] for x in imgclass_data] # Familiar Labels
eventlabel_deviant = [imgclass_data[x]['EventlabelDeviant'] for x in imgclass_data] # Deviant Labels
class_sets =  [imgclass_data[x]['ClassSet'] for x in imgclass_data] # Load classes (subset of images used)
nimg = 8   

# set SVM type to use (Linear results shown in paper)
svm_type = 'linear' # Option for nonlinear or logistic if needed

# Set up results to save across all sessions used, saving ROC AUC information
roc_out = {k: np.zeros(nsess) for k in keys_firing} # Save average ROC AUC per session
stdroc_out = {k: np.zeros(nsess) for k in keys_firing} # Save std ROC AUC per sessions
rocShuff_out = {k: np.zeros(nsess) for k in keys_firing} # Save Shuffled ROC AUC per serssion
evcounts_out = np.zeros((nsess,nimg))  # Count number of images per session
roc_curves_out = {k: [] for k in keys_firing}
pred_out = [] # Store predicted labels
nTs = np.zeros(nsess) # Number of sessions 

print(f'Behavior: Fitting with {svm_type} over {nsess} sessions')
for zz in trange(nsess):
    countsT_ = counts_deviantT[zz] # Tonic Counts
    countsB_ = counts_deviantB[zz] # Burst Counts
    eventlabels_ = eventlabel_deviant[zz] # Behavior Labels
    imglabel_ = imglabels_behave[zz] # Img Labels
    
    _, eventdeviant_counts = np.unique(eventlabels_, return_counts=True) # Number of each img
    evcounts_out[zz,:] = eventdeviant_counts # Save the counts per image
    min_deviant = np.min(eventdeviant_counts) # Min deviant count - will be used to equalize training set
    eventlabel_deviant_sub = np.repeat(range(nimg),min_deviant) # Set up eventlabels
    ndeviant = len(eventlabel_deviant_sub) # Number of deviants
    
    nN = countsT_.shape[1] # Number of neurons
    nTs[zz] = nN # Save number of neurons per trial
    
    # Build LDA; One for tonic and one for burst. Results set up to get ROC info out
    lda_tonic_roc = build_lda(countsT_,eventlabels_,svm_type,metric_type='roc') # Tonic ROC 
    lda_burst_roc = build_lda(countsB_,eventlabels_,svm_type,metric_type='roc') # Burst ROC

    # --- Subsample our data nsub times and calc ROC AUC each time ---
    # Note, we are only carring about deviant -> deviant for tonic and burst
    # Also save shuffle results
    roc_data_ = {k: np.zeros(nsub) for k in keys_firing}
    roc_data_shuff_ = {k: np.zeros(nsub) for k in keys_firing}
    roc_curves_ = {k: [] for k in keys_firing}
    preds = []
    for ns in range(nsub):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=ns) # stratified shuffled
        train_idx, test_idx = next(sss.split(countsT_, eventlabels_)) # Train and test index for tonic (same as burst)

        # Fixed test set as to keep true population proportions
        countsT_test = countsT_[test_idx] # Tonic Test
        countsB_test = countsB_[test_idx] # Burst Test
        eventlabel_test = eventlabels_[test_idx] # Label Test
        
        # --- Set training loop and equalize image presentations ---
        countsT_train = countsT_[train_idx] # Tonic Training set
        countsB_train = countsB_[train_idx] # Burst Training set
        eventlabel_train = eventlabels_[train_idx] # Label train
        
        # Equalize training only
        _, train_counts = np.unique(eventlabel_train, return_counts=True) # Get counts per image
        min_train = np.min(train_counts) # Take min number as equalizer
        
        countsT_train_sub = np.zeros((min_train * nimg, nN))
        countsB_train_sub = np.zeros((min_train * nimg, nN))
        for i in range(nimg):
            idx = np.where(eventlabel_train == i)[0] # Find ith images
            idx_sub = np.random.permutation(len(idx))[:min_train] # Randomize the subset
            countsT_train_sub[i*min_train:(i+1)*min_train] = countsT_train[idx[idx_sub]] # Tonice subset
            countsB_train_sub[i*min_train:(i+1)*min_train] = countsB_train[idx[idx_sub]] # Burst subset
        
        eventlabel_train_sub = np.repeat(range(nimg), min_train) # Label Subset
        
        # Store true and predicted labels
        y_pred_ = np.zeros(((len(countsB_test)),3)) 
        y_pred_[:,0] = eventlabel_test # True labels
        
        
        # --- FIT AND GET ROC INFO ---
        roc_data_['Tonic'][ns], _, curves_h_T, _ = run_roc_per_image(lda_tonic_roc,countsT_train_sub, 
                                                                     countsT_test, eventlabel_train_sub,
                                                                     eventlabel_test) # Tonic ROC
        roc_curves_['Tonic'].append(curves_h_T) # Tonic ROC curve

        roc_data_['Burst'][ns], _, curves_h_B, _ = run_roc_per_image(lda_burst_roc, countsB_train_sub, 
                                                                     countsB_test, eventlabel_train_sub,
                                                                     eventlabel_test) # Burst ROC    
        roc_curves_['Burst'].append(curves_h_B) # Burst ROC curve

        # Shuffle controls
        roc_data_shuff_['Tonic'][ns] = run_roc_shuffle(lda_tonic_roc,countsT_train_sub, countsT_test,  
                                                       eventlabel_train_sub,eventlabel_test, n_classes=nimg)
        roc_data_shuff_['Burst'][ns] = run_roc_shuffle(lda_burst_roc,countsB_train_sub, countsB_test, 
                                                       eventlabel_train_sub,eventlabel_test, n_classes=nimg)
        preds.append(y_pred_)

    pred_out.append(preds) # Get predictions for each sessions

    # Average curves across shuffles for given session
    mean_fpr = np.linspace(0, 1, 100)
    for k in keys_firing:
        roc_out[k][zz] = np.mean(roc_data_[k])
        stdroc_out[k][zz] = np.std(roc_data_[k])
        rocShuff_out[k][zz] = np.mean(roc_data_shuff_[k])
        
        # Average and SE ROC curves
        auc_per_img = np.zeros((nsub, nimg))
        tpr_per_img = np.zeros((nsub, nimg, 100))
        for nn in range(nsub):
            auc_per_img[nn] = roc_curves_[k][nn]['auc']
            for ni in range(nimg):
                tpr_per_img[nn, ni] = np.interp(mean_fpr,
                    roc_curves_[k][nn]['fpr'][ni],
                    roc_curves_[k][nn]['tpr'][ni])
        roc_curves_out[k].append({
            'mean_auc': np.mean(auc_per_img, axis=0),
            'mean_tpr': np.mean(tpr_per_img, axis=0),
            'sem_tpr': np.std(tpr_per_img, axis=0) / np.sqrt(nsub),
            'mean_fpr': mean_fpr
        })
        
#%% Plot ROC results for images
# --- TONIC | BURST ROC Box Plot ---
# Box plot parameters
_, p_ = st.wilcoxon(roc_out['Tonic'],roc_out['Burst'])
_, pTbase_ = st.wilcoxon(roc_out['Tonic'],rocShuff_out['Tonic'], alternative='greater')
_, pBbase_ = st.wilcoxon(roc_out['Burst'],rocShuff_out['Burst'], alternative='greater')

pos_ = [0,0.3,0.9,1.2] # Positions 
cls_ = [pltcolors[0],'grey',pltcolors[1],'grey'] # Colors
lbs_ = ['Test familiar','Shuffle','Test deviant','Shuffle'] # Labels

roc_together = [roc_out['Tonic'],rocShuff_out['Tonic'],
                roc_out['Burst'],rocShuff_out['Burst']] # Pack data together
rs = np.random.randn(4,nsess)*0.02 # Jitter for xlabels

figbox = plt.figure(2); plt.clf()
# Box plot
plt.boxplot(roc_together,positions=pos_,widths=0.3,showfliers=False,whis=[0, 100]) # Box plots
for i in range(len(pos_)):
    plt.scatter(pos_[i]*np.ones(nsess)+rs[i],roc_together[i],
                color=cls_[i],label=lbs_[i]) # Tonic Scatter
plt.ylabel('ROC AUC')
ax = plt.gca()
# Add sigs
boxsig(p_, pos_[0], pos_[2], 1.1, 0.05, ax)
boxsig(pTbase_, pos_[0], pos_[1], 1, 0.05, ax)
boxsig(pBbase_, pos_[2], pos_[3], 1, 0.05, ax)
# Add connecting lines
for i in range(nsess):
    plt.plot([rs[0][i],pos_[2]+rs[2][i]],[roc_together[0][i],roc_together[2][i]],'grey',alpha=0.3)

chance = 0.5 
width = 0.3

plt.ylim([0,1.5]);
plt.hlines(chance,pos_[0]-width,pos_[-1]+width,'k',linestyles='--')
plt.xticks([0.15,1.05],['Tonic','Burst']);
plt.xlim([pos_[0]-width,pos_[-1]+width])
plt.tight_layout()

# --- ROC vs Number of neurons ---
# Fit lines for Tonic and Burst on N counts vs ROC
xx = np.arange(0,110,5)
pT = np.polyfit(nTs,roc_out['Tonic'],1) 
yyT = np.polyval(pT,xx)
rT, pT = st.pearsonr(nTs, roc_out['Tonic']) # Get corr and p-value 

pB = np.polyfit(nTs,roc_out['Burst'],1)
yyB = np.polyval(pB,xx)
rB, pB = st.pearsonr(nTs, roc_out['Burst']) # Get corr and p-value 

# Confidence intetervals
ci_T = st.t.ppf(0.975, df=nsub-1) * (stdroc_out['Tonic'] / np.sqrt(nsub))
ci_B = st.t.ppf(0.975, df=nsub-1) * (stdroc_out['Burst'] / np.sqrt(nsub))

figN = plt.figure(3); plt.clf()
plt.errorbar(nTs-0.1, roc_out['Tonic'], yerr = ci_T,fmt='o',capsize=5,label='Tonic')
plt.errorbar(nTs+0.1, roc_out['Burst'], yerr = ci_B,fmt='o',capsize=5,label='Burst')
plt.plot(xx,yyT,color=pltcolors[0],linestyle='--')
plt.plot(xx,yyB,color=pltcolors[1],linestyle='--')
plt.xticks(xx)
plt.legend()
plt.grid(True)
plt.xlim([0,100])
plt.ylim([0.4,1])
plt.xlabel('N'); plt.ylabel('ROC AUC')
plt.hlines(chance,0,100,'k',linestyles='--')
plt.tight_layout()

# --- Average ROC AUC for each image ---
ncb = np.sum(np.array(class_sets) == 'classB') # Only do the analysis on Class B sets
tonic_mean_auc = np.zeros((ncb,nimg)) 
burst_mean_auc = np.zeros((ncb,nimg))
flag = 0 # Flag for storing
for i in range(nsess):
    if class_sets[i] == 'classA': continue # Ignore class A images
    session_dataT = roc_curves_out['Tonic'][flag] # Tonic ROC curves
    session_dataB = roc_curves_out['Burst'][flag] # Burst ROC curves
    for j in range(nimg):
        tonic_mean_auc[flag,j] = session_dataT["mean_auc"][j] # Tonic ROC for img j
        burst_mean_auc[flag,j] = session_dataB["mean_auc"][j] # Burst ROC for img j
    flag+=1 
    
# Tonic ROC stats
mean_meanT = np.mean(tonic_mean_auc,0)
se_meanT = np.std(tonic_mean_auc,0)/np.sqrt(nsess)

# Burst ROC stats
mean_meanB = np.mean(burst_mean_auc,0)
se_meanB = np.std(burst_mean_auc,0)/np.sqrt(nsess)

xx = np.arange(1,nimg+1)
widths = 0.3

plt.figure(3); plt.clf()
plt.bar(xx-widths/2,mean_meanT,yerr = se_meanT,capsize=5,width=widths,label='Tonic')
plt.bar(xx+widths/2,mean_meanB,yerr = se_meanB,capsize=5,width=widths,label='Burst')
plt.legend()
plt.xticks(xx)
plt.ylabel('ROC AC')
plt.xlabel('Image')
plt.tight_layout()

