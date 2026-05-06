from allensdk.core.reference_space_cache import ReferenceSpaceCache
import re
import nrrd
import os
import requests

class VBN_CCF:

    def __init__(self, manifest_path='manifest.json', resolution=10):
        reference_space_key = 'annotation/ccf_2017'
        self.resolution = resolution
        self.rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=manifest_path)
        # ID 1 is the adult mouse structure graph
        self.tree = self.rspc.get_structure_tree(structure_graph_id=1) 
        self._annotation = None
        self._streamlines = None
        self._manifest_directory = os.path.dirname(manifest_path)


    @property
    def annotation(self):
        if self._annotation is None:
            self._annotation, meta = self.rspc.get_annotation_volume()
        return self._annotation

    
    @property
    def streamlines(self):
        if self._streamlines is None:
            streamlines_path = os.path.join(self._manifest_directory, 'laplacian_10.nrrd')
            #First check to see if the streamlines have already been downloaded
            if os.path.exists(streamlines_path):
                self._streamlines, header = nrrd.read(streamlines_path)
            
            #Otherwise download it and write it to file
            else:
                s = requests.get('https://www.dropbox.com/sh/7me5sdmyt5wcxwu/AACFY9PQ6c79AiTsP8naYZUoa/laplacian_10.nrrd?dl=1')
                with open(streamlines_path, 'wb') as f:
                    f.write(bytes(s.content))
                self._streamlines, header = nrrd.read(streamlines_path)
                
        return self._streamlines


    def get_structure_by_acronym(self, acronym):
        try:
            structure = self.tree.get_structures_by_acronym([acronym])
        except KeyError:
            print(f'Could not find structure corresponding to acronym {acronym}')
            structure = [{}]
        return structure


    def get_structure_name_by_acronym(self, acronym):
        structure = self.get_structure_by_acronym(acronym)[0]
        return structure.get('name', None)

    
    def get_structure_id_by_coordinate(self, ap_coord, dv_coord, lr_coord):

        volume_coords = [int(coord/self.resolution) for coord in [ap_coord, dv_coord, lr_coord]]
        shape = self.annotation.shape
        if any([(v<0 or v>=s) for v,s in zip(volume_coords,shape)]):
            print(f'Coordinate {[ap_coord, dv_coord, lr_coord]} is outside ccf')
            id = 0
        else:
            id = self.annotation[volume_coords[0], volume_coords[1], volume_coords[2]]
        return id


    def get_structure_by_id(self, id):
        if id == 0:
            structure = [{'name': 'outside_brain',
                        'acronym': 'outside_brain'}]
        else:
            structure = self.tree.get_structures_by_id([id])
        return structure


    def get_structure_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        try:
            id = self.get_structure_id_by_coordinate(ap_coord, dv_coord, lr_coord)
            structure = self.get_structure_by_id(id)
        except Exception as e:
            print(f'Could not get structure corresponding to id: {id} due to error {e}')
            structure = [{}]
        return structure


    def get_structure_acronym_by_coordinate(self, ap_coord, dv_coord, lr_coord):

        structure = self.get_structure_by_coordinate(ap_coord, dv_coord, lr_coord)[0]
        return structure.get('acronym', None)

    
    def get_cortical_layer_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        
        acronym = self.get_structure_acronym_by_coordinate(ap_coord, dv_coord, lr_coord)
        return self.get_cortical_layer_name_by_acronym(acronym)
        
        
    def get_cortical_layer_name_by_acronym(self, acronym):

        if acronym in ['CA1', 'CA2', 'CA3']:
            return ''
    
        try:
            first_num = re.findall(r'\d+', acronym)[0]
            first_num_ind = acronym.find(first_num)
            if first_num_ind<0:
                return ''
            
            layer = acronym[first_num_ind:]
            return layer

        except IndexError:
            return ''
    

    def get_cortical_depth_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        volume_coords = [int(coord/10) for coord in [ap_coord, dv_coord, lr_coord]]
        try:
            cortical_depth = self.streamlines[volume_coords[0], volume_coords[1], volume_coords[2]]
        except IndexError as e:
            print(f'Coordinate {[ap_coord, dv_coord, lr_coord]} is outside of CCF')
            cortical_depth = 0
        return cortical_depth


# ---------------------------


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #Useful tool to arrange multiple plots in one figure (https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html)
from pathlib import Path

%matplotlib inline

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

from brain_observatory_utilities.datasets.electrophysiology.receptive_field_mapping import ReceptiveFieldMapping_VBN

#instatiate tool: this will automatically download the small files (manifest and structure jsons)
c = VBN_CCF()

data_root = r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project'

output_dir = r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Raw_Data'
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=Path(output_dir))

unit_dir =  Path ( r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Extracted_Data\units' ) 
sessions_dir = Path(r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Extracted_Data\session_data')
lfp_dir = Path ( r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Extracted_Data\lfp' ) 
meta_data_fldr = r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Raw_Data\visual-behavior-neuropixels-0.5.0\project_metadata'
nwb_path = Path ( r'C:\Users\za23e\OneDrive - Florida State University\AllenNP\Raw_Data\visual-behavior-neuropixels-0.5.0\behavior_ecephys_sessions' ) 



# ---------------------------------------------


ecephys_sessions_table = cache.get_ecephys_session_table()
# ecephys_sessions_table.head()

sessions_to_keep = ecephys_sessions_table[
                    (ecephys_sessions_table.experience_level == 'Novel')&                   
                    (ecephys_sessions_table.structure_acronyms.str.contains('LGd'))]

# sessions_to_keep
all_valid_sessions = sessions_to_keep.index
len(all_valid_sessions)
print (all_valid_sessions)

session_list = all_valid_sessions


final_session_list = [1048196054, 1053925378, 1063010385, 1064639378, 1065908084,
                     1067781390, 1081431006, 1086410738, 1091039376, 1092466205,
                     1096935816, 1104297538, 1108528422, 1108531612, 1112515874, 
                     1115356973, 1120251466, 1121607504, 1122903357, 1130349290]


# --------------------------------------------------------
# Session level data 
# -------------------------------------------------------------



import h5py

# Get session information - the different trial type times and pupil diameter 
def find_nearest(lst, value):
    return min(lst, key=lambda x: abs(x - value))

for session_id in final_session_list: 

    sessionoutputpath = sessions_dir / f'session_{session_id}'
    isExist = os.path.exists(sessionoutputpath)
    if not isExist:
        os.makedirs(sessionoutputpath)
        print("The new directory was created!")
    os.chdir(sessionoutputpath)

    print(f'working with {session_id}' ) 

    session = cache.get_ecephys_session(ecephys_session_id=session_id)

    trials = session.trials
    stims = session.stimulus_presentations

    hit_go_trials = stims[stims['is_change']==True]
    hit_go_trials = hit_go_trials[hit_go_trials['rewarded']==True]

    all_start_times = stims.start_time

    # CONFIRMED THAT THE START_TIME ON THE STIM PRSENATIONS IS WHAT WE NEED (https://allensdk.readthedocs.io/en/latest/_static/examples/nb/aligning_behavioral_data_to_task_events_with_the_stimulus_and_trials_tables.html#Introduction-to-the-stimulus-presentations-table)
    hit_go_trials = stims[stims['is_change']==True]                             # all GO trials
    hit_go_trials = hit_go_trials[hit_go_trials['rewarded']==True]              # that were a hit
    hit_go_trials = hit_go_trials[hit_go_trials['stimulus_block']==0]           # during the ACTIVE period
    hit_go_trials = hit_go_trials[3:]                                           # the first three trials are autorewarded so reward = true even with no response + a larger volume that usual so ignore
    hit_go_trials = hit_go_trials['start_time'].values
    hit_go_trials


    # GET THE CORRECT REJECT TRIALS                                 
    correct_reject_trials = trials [ trials ['correct_reject'] == True]                     # have to use the trials data because the stims data just gives whether sham_change = true but not whether it was correcty rejected. 
    correct_reject_trials = correct_reject_trials['change_time_no_display_delay']   


    correct_reject_start_times=[]
    for st in correct_reject_trials.index:
        nearest = find_nearest(all_start_times, correct_reject_trials[st])
        n_df = np.float32([nearest])
        correct_reject_start_times= np.concatenate((correct_reject_start_times, n_df), axis=0)



    # PASSIVE START TIMES 
    changed_images_passive = stims[ (stims['active']==False )  & 
                                            (stims['is_change']==True) ]
    changed_images_passive_start_times = np.array(changed_images_passive.start_time)


    # MISSED GO   
    missed_go = stims[stims['is_change']==True]                             # all GO trials
    missed_go = missed_go[missed_go['rewarded']==False]              # that were a hit
    missed_go = missed_go[missed_go['stimulus_block']==0]           # during the ACTIVE period                                
    missed_go = missed_go['start_time'].values


    # FALSE ALARM - A CATCH IN WHICH THERE WAS A LICK 
    false_alarm_trials = trials [ trials ['false_alarm'] == True]
    false_alarm_trials = false_alarm_trials['change_time_no_display_delay']   

    false_alarm_trials_start_times=[]
    for fa in false_alarm_trials.index:
        nearest = find_nearest(all_start_times, false_alarm_trials[fa])
        n_df = np.float32([nearest])
        false_alarm_trials_start_times= np.concatenate((false_alarm_trials_start_times, n_df), axis=0)



    # ABORTED TRIALS 
    aborted = trials [ trials ['aborted'] == True]
    aborted = aborted['change_time_no_display_delay']   

    aborted_start_times=[]
    for ab in aborted.index:
        nearest = find_nearest(all_start_times, aborted[ab])
        n_df = np.float32([nearest])
        aborted_start_times= np.concatenate((aborted_start_times, n_df), axis=0)


    # TRUE NOVEL - DEFINED HERE AS THE FIRST TIME A MOUSE SAW A PARTICULAR IMAGE
    im_names = stims["image_name"].unique().tolist()
    im_names

    true_novel_trials = pd.DataFrame(columns=stims.columns) 

    for i in im_names:
        if i != "ommited" or pd.notna(i):
            ii = stims[ (stims.image_name == i) & 
                    (stims.is_image_novel == True) &
                     (stims.is_change == True) & 
                      (stims.rewarded == True) &
                      (stims.active == True )]

            if ii.empty:
                continue 

            trial_oi = ii.iloc[[0]]
            true_novel_trials = pd.concat([true_novel_trials, trial_oi], ignore_index=True)

    true_novel_trials_start_times  = true_novel_trials.start_time 
    true_novel_trials_start_times = np.array(true_novel_trials_start_times)   


    # FREE REWARD TRIALS  
    free_reward_trials = trials [trials ['auto_rewarded'] == True ]
    free_reward_trials = free_reward_trials['change_time_no_display_delay']   

    free_reward_trials_start_times=[]

    for fa in free_reward_trials.index:
        nearest = find_nearest(all_start_times, free_reward_trials[fa])
        n_df = np.float32([nearest])
        free_reward_trials_start_times= np.concatenate((free_reward_trials_start_times, n_df), axis=0)



        # --- Your arrays (make sure each is a numpy array, not a list/Series) ---
        hit_go_trials_start_times        = np.array(hit_go_trials)
        correct_reject_start_times       = np.array(correct_reject_start_times)
        changed_images_passive_start_times = np.array(changed_images_passive_start_times)
        missed_go_start_times            = np.array(missed_go)
        false_alarm_trials_start_times   = np.array(false_alarm_trials_start_times)
        aborted_start_times              = np.array(aborted_start_times)
        true_novel_trials_start_times    = np.array(true_novel_trials_start_times)
        free_reward_trials_start_times   = np.array(free_reward_trials_start_times)

        # --- Put them in a dictionary for convenience ---
        trial_dict = {
            "hit_go": hit_go_trials_start_times,
            "correct_reject": correct_reject_start_times,
            "passive": changed_images_passive_start_times,
            "missed_go": missed_go_start_times,
            "false_alarm": false_alarm_trials_start_times,
            "aborted": aborted_start_times,
            "true_novel": true_novel_trials_start_times,
            "free_reward": free_reward_trials_start_times,
            }

        # --- Session-specific filename ---
        h5_path = sessionoutputpath / f"session_{session_id}_trial_start_times.h5"

        with h5py.File(h5_path, "w") as h5f:
            for key, arr in trial_dict.items():
                h5f.create_dataset(key, data=arr)  # no compression




    
    # GET EYE INFORMATION
    
    eye_tracking = session.eye_tracking

    pupil_area = eye_tracking.pupil_area
    pupil_area = np.array(pupil_area)

    timestamps = eye_tracking.timestamps 
    ts = np.array(timestamps)

    ts_pupilsize = np.vstack( (ts, pupil_area)) 

    pupil_XY = np.vstack((
        np.array(eye_tracking['pupil_center_x']), 
        np.array(eye_tracking['pupil_center_y']), 
        np.array(eye_tracking['timestamps'])
    ))

    np.save(f'session{session_id}_ts_pupil_area.npy', ts_pupilsize)
    np.save(f'session{session_id}_pupil_XY.npy', pupil_XY)

# --------------------------------------------------------
# Unit  level data 
# -------------------------------------------------------------


## EXTRACT ALL UNITS FROM A SINGLE AREA.

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

mcc = MouseConnectivityCache(resolution=25)
rsp = mcc.get_reference_space()

sg = rsp.remove_unassigned()
sg = pd.DataFrame(sg)

import os
import numpy as np
import pandas as pd
import h5py

# get units based on isi_violations and receptive field properties and location in mask 
os.chdir(meta_data_fldr)
ALL_units = pd.read_csv('units.csv')

roi_ac_list = ['LGd'] 

for session_id in final_session_list: 

    print(session_id)

    ephys_session = cache.get_ecephys_session(ecephys_session_id=session_id)

    units = ephys_session.get_units(amplitude_cutoff_maximum=np.inf,
                                    presence_ratio_minimum=-np.inf)

    channels = ephys_session.get_channels()
    units = units.merge(channels, left_on='peak_channel_id', right_index=True)

    # create session folder
    sessionoutputpath = unit_dir / f'session_{session_id}'
    # os.makedirs(sessionoutputpath, exist_ok=True)

    for roi_ac in roi_ac_list:


        sg = sg[sg['acronym'] == (roi_ac)]

        roi_id = int (sg.id ) 
        print(roi_id)
        roi_mask = rsp.make_structure_mask([roi_id], direct_only=False) 

    
        ROI_units = units[(units['structure_acronym']==roi_ac) & 
                          (units['quality'] == 'good') & 
                          (units['isi_violations'] < 0.5)]              

        uids = ROI_units.index.tolist()


        if roi_ac == 'LGd':

            rf = ReceptiveFieldMapping_VBN(ephys_session, filter=uids)
            rf_metrics = rf.metrics

            units_with_rfstats = ROI_units.merge(rf_metrics, left_index=True, right_on='unit_id')

            good_units = units_with_rfstats[(units_with_rfstats.on_screen_rf==True) &
                                        (units_with_rfstats.p_value_rf<0.01)]
            
        else:             
            good_units = ROI_units


        print(len(good_units))

        # check CCF mask
        final_good_unit_list = [] 
        for i in good_units.index.tolist():
            CCF = np.hstack((good_units['anterior_posterior_ccf_coordinate'].loc[i], 
                            good_units['dorsal_ventral_ccf_coordinate'].loc[i], 
                            good_units['left_right_ccf_coordinate'].loc[i]))  
            CCF = CCF // 25
            if roi_mask[int(CCF[0]), int(CCF[1]), int(CCF[2])] == 1:
                final_good_unit_list.append(i)   

        print(len(good_units) - len(final_good_unit_list))    

        spike_times = ephys_session.spike_times
        


        # ---------------------------------------
        # Create HDF5 file for this session + ROI of unit times and metadata
        # ---------------------------------------
        # h5_path = sessionoutputpath / f"session_{session_id}_{roi_ac}_units.h5"
        
        # with h5py.File(h5_path, "w") as f:

        #     # Session-level info
        #     f.create_dataset("session_id", data=int(session_id))
        #     f.create_dataset("region", data=str(roi_ac))

        #     # Group for units
        #     units_grp = f.create_group("units")

        #     for u in final_good_unit_list: 
        #         ugrp = units_grp.create_group(f"unit{u}")

        #         ugrp.create_dataset("id", data=int(u))

        #         # spike times
        #         spikes = np.array(spike_times[u])
        #         ugrp.create_dataset("spike_times", data=spikes)

        #         # coordinates
        #         coords = np.array([
        #             good_units.loc[u, 'anterior_posterior_ccf_coordinate'],
        #             good_units.loc[u, 'left_right_ccf_coordinate'],
        #             good_units.loc[u, 'dorsal_ventral_ccf_coordinate']
        #         ])
        #         ugrp.create_dataset("Coords_AP_ML_DV", data=coords)

        #         # waveform features
        #         ind_u = ALL_units.loc[ALL_units['unit_id'] == u].index
                
        #         ugrp.create_dataset("amplitude", data=float(ALL_units.amplitude[ind_u]))
        #         ugrp.create_dataset("halfwidth", data=float(ALL_units.waveform_halfwidth[ind_u]))
        #         ugrp.create_dataset("pt_ratio", data=float(ALL_units.PT_ratio[ind_u]))
        #         ugrp.create_dataset("waveform_duration", data=float(ALL_units.waveform_duration[ind_u]))

        #         ugrp.create_dataset("layer", data=str('n/a'))



        # ---------------------------------------
        # Create HDF5 file for this session + ROI of unit quality metrics 
        # ---------------------------------------
        h5_path = sessionoutputpath / f"session_{session_id}_{roi_ac}_unit_quality_metrics.h5"
        print('h5_path')
        
        with h5py.File(h5_path, "w") as f:

            # Session-level info
            f.create_dataset("session_id", data=int(session_id))
            f.create_dataset("region", data=str(roi_ac))

            # Group for units
            units_grp = f.create_group("units")

            for u in final_good_unit_list: 

                ind_u = ALL_units.loc[ALL_units['unit_id'] == u].index
                ugrp = units_grp.create_group(f"unit{u}")

                ugrp.create_dataset("id", data=int(u))
                ugrp.create_dataset("ISI_ratio", data=float(ALL_units.isi_violations[ind_u]))
                ugrp.create_dataset("presence_ratio", data=float(ALL_units.presence_ratio[ind_u]))
                ugrp.create_dataset("snr", data=float(ALL_units.snr[ind_u]))



