import os
from pathlib import Path

import pandas as pd
import numpy as np
import json

from scipy.interpolate import RegularGridInterpolator, griddata
from scipy import signal
from scipy.ndimage.filters import gaussian_filter


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #Useful tool to arrange multiple plots in one figure (https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html)

import quantities as pq

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

from brain_observatory_utilities.datasets.electrophysiology.receptive_field_mapping import ReceptiveFieldMapping_VBN

from typing import List, Tuple



# From Allen databook [X]
def select_good_channels(lfp: np.ndarray,
                         reference_channels: List[int],
                         noisy_channel_threshold: float
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove reference channels and channels that are too noisy from lfp data.

    Parameters
    ----------
    lfp : numpy.ndarray
        LFP data in the form of: trials x channels x time samples
    reference_channels : List[int]
        Reference channel indices for this probe.
    noisy_channel_threshold : float
        Lowest mean standard deviation that constitutes a "clean" LFP channel

    Returns
    -------
    Tuple[cleaned_lfp, good_indices]
        cleaned_lfp: numpy.ndarray
            LFP where reference and noisy channels have been removed.
            Data still in form of: trials x channel x time samples
        good_indices: numpy.ndarray
            Array of channel indices that are neither reference nor noisy.
    """
    channel_variance = np.mean(np.std(lfp, 2), 0)
    noisy_channels = np.where(channel_variance > noisy_channel_threshold)[0]

    to_remove = np.concatenate(
        (
            np.array(reference_channels),
            noisy_channels
        )
    ).astype(int)
    good_indices = np.delete(np.arange(0, lfp.shape[1]), to_remove)

    # Remove noisy or reference channels (axis=1)
    cleaned_lfp = np.delete(lfp, to_remove, axis=1)

    return (cleaned_lfp, good_indices)


def filter_lfp_channels(lfp: np.ndarray,
                        sampling_rate: float,
                        filter_cuts: List[float],
                        filter_order: int) -> np.ndarray:
    '''Bandpass filter lfp channel data.

    Parameters
    ----------
    lfp : numpy.ndarray
        LFP data to be filtered in the form of:
        trials x channels x time samples
    sampling_rate : float
        Sampling rate for lfp data
    filter_cuts : List[float]
        Low and high cut for bandpass filter
    filter_order : int
        Order for bandpass filter

    Returns
    -------
    filtered_lfp: numpy.ndarray
        LFP that has been bandpassed filtered along the sample axis.
        Still in the form of: trials x channels x time samples
    '''

    wn = (sampling_rate / 2)
    filter_cutoffs = np.array(filter_cuts) / wn
    b, a = signal.butter(filter_order, filter_cutoffs, 'bandpass')
    # Bandpass filter time samples (axis=2)
    filtered_lfp = signal.filtfilt(b, a, lfp, axis=2)

    return filtered_lfp


def make_actual_channel_locations(min_chan: int = 0,
                                  max_chan: int = 384) -> np.ndarray:
    '''Generate x/y locations of Neuropixels recording sites.

          0  8 16 24 32 40 48
    60    *  -  -  -  *  -  -
    50    -  -  -  -  -  -  -
    40    -  -  *  -  -  -  * <-- actual recording site (*)
    30    -  -  -  -  -  -  -
    20    *  -  -  -  *  -  -
    10    -  -  -  -  -  -  -
     0    -  -  *  -  -  -  *

    Parameters
    ----------
    min_chan : int, optional
        Lowest channel number to use, by default 0
    max_chan : int, optional
        Highest channel number to use, by default 384

    Returns
    -------
    actual_channel_locations: numpy.ndarray
        column 1 = x positions in microns
        column 2 = y positions in microns
    '''

    actual_channel_locations = np.zeros((max_chan, 2))

    x_locations = [16, 48, 0, 32]

    for ch in range(min_chan, max_chan):
        actual_channel_locations[ch, 0] = x_locations[ch % 4]
        actual_channel_locations[ch, 1] = np.floor(ch / 2) * 20

    return actual_channel_locations[min_chan:, :]


def make_interp_channel_locations(min_chan: int = 0,
                                  max_chan: int = 384) -> np.ndarray:
    '''Generate x/y locations for interpolated Neuropixels recording sites.

    This version just returns the central column of interpolated sites.

          0  8 16 24 32 40 48
    60    *  -  -  o  *  -  -
    50    -  -  -  o  -  -  -
    40    -  -  *  o  -  -  * <-- actual recording site (*)
    30    -  -  -  o  -  -  -
    20    *  -  -  o  *  -  -
    10    -  -  -  o  -  -  -
     0    -  -  *  o  -  -  *
                   ^
                   interpolated column sites (o)

    Parameters
    ----------
    min_chan : int, optional
        Lowest channel number to use, by default 0
    max_chan : int, optional
        Highest channel number to use, by default 384

    Returns
    -------
    interp_channel_locations: numpy.ndarray
        column 1 = interpolated x positions in microns
        column 2 = y positions in microns
    '''

    interp_channel_locations = np.zeros((max_chan, 2))

    for ch in range(min_chan, max_chan):
        interp_channel_locations[ch, 0] = 24
        interp_channel_locations[ch, 1] = ch * 10

    return interp_channel_locations[min_chan:, :]


def interp_channel_locs(lfp: np.ndarray,
                        actual_locs: np.ndarray,
                        interp_locs: np.ndarray,
                        method: str = 'linear') -> Tuple[np.ndarray, float]:
    '''Interpolates single-trial lfp channel locations to account for
    channel stagger.

    Parameters
    ----------
    lfp : numpy.ndarray
        LFP data in the form of: trials x channels x time samples
    actual_locs: numpy.ndarray
        An array of actual x, y locations for all channels in lfp. The
        number of actual_locs should equal the number of channels in the 'lfp'.
    interp_locs: numpy.ndarray
        An array of virtual x, y locations for where channels in lfp
        should be interpolated to.

    method : str, optional
        Interpolation method ['cubic', 'linear', 'nearest'], by default 'cubic'

    Returns
    -------
    Tuple[interp_lfp, spacing]
        interp_lfp: numpy.ndarray
            Channel location interpolated lfp data in the form of:
            trials x channels x time samples
        spacing: float
            Distance between new interpolated virtual channel sites
            (in millimeters)
    '''

    if lfp.shape[1] != actual_locs.shape[0]:
        e_msg = (f"Number of 'lfp' channels ({lfp.shape[1]}) does not "
                 f"match number of 'actual_locs' ({actual_locs.shape[0]})!")
        raise RuntimeError(e_msg)

    spacing = np.mean(np.diff(interp_locs[:, 1])) / 1000

    interp_lfp = np.zeros((lfp.shape[0],  # number of interp trials
                           interp_locs.shape[0],  # number of interp channels
                           lfp.shape[2]))  # number of interp samples

    for trial in range(lfp.shape[0]):  # trials
        trial_data = lfp[trial, :, :]
        for t in range(0, lfp.shape[2]):  # time samples
            interp_lfp[trial, :, t] = griddata(points=actual_locs,
                                               values=trial_data[:, t],
                                               xi=interp_locs,
                                               method=method,
                                               fill_value=0,
                                               rescale=False)

    return (interp_lfp, spacing)




def display_response_window(window, start_time, end_time, vmin=None, vmax=None, title="", xlabel="", ylabel="", cbar_label=""):
    fig, ax = plt.subplots(figsize=(6,6))

    img = ax.imshow(window, 
                    extent=[start_time, end_time, 3840, 0], # probe is 3840 micrometers 
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax
                ) 

    # make dotted line at stimulus onset
    ax.plot([0,0],[0, 3840], ':', color='white', linewidth=1.0)

    cbar = fig.colorbar(img, shrink=0.5)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def regular_grid_extractor_factory(timestamps: np.ndarray,
                                   lfp_raw: np.ndarray,
                                   channel: int,
                                   method: str = 'linear') -> np.ndarray:
    '''Builds an LFP data extractor using interpolation on a regular grid

    Ignores timestamps less than zero (which result from unaligned
    data segments)

    Parameters
    ----------
    timestamps : numpy.ndarray
        Associates LFP sample indices with times in seconds.
    lfp_raw : numpy.ndarray
        Dimensions are samples X channels.
    channel : int
        Index of channel to interpolate to regular grid.
    method : str, optional
        Interpolation method ['linear', 'cubic', 'nearest'],
        by default 'linear'.

    Returns
    -------
    numpy.ndarray
        LFP data that has been interpolated to a regular grid.
    '''

    valid_timestamps = (timestamps >= 0)

    return RegularGridInterpolator((timestamps[valid_timestamps],),
                                   lfp_raw[valid_timestamps, channel],
                                   method=method,
                                   bounds_error=False,
                                   fill_value=np.nan)


# --------------
# --------------
# --------------


output_dir = r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Raw_Data'
lfp_dir = Path ( r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Extracted_Data\lfp_band' ) 

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=Path(output_dir))


session_list = [1048196054, 1053925378, 1063010385, 1064639378, 1065908084,
                1067781390, 1081431006, 1091039376, 1092466205,
                1096935816, 1104297538, 1108528422, 1108531612, 1112515874, 
                1115356973, 1121607504,  1130349290]


# REMOVED - RAW DATA NOT UPLOADED FOR 1120251466, 1122903357
# ALSO REMOVED Because VISp LFP not available for session 1086410738


# ----------------------------------------------------------
# FIRST GET LFP AS trials x channels x time samples
# ----------------------------------------------------------
def find_nearest(lst, value):
    return min(lst, key=lambda x: abs(x - value))

window_start_time = -0.1
window_end_time = 0.75

os.chdir(lfp_dir)
dir_to_save = (r'Z:\Zoe\Allen_Institute_Project\Extracted_Data\lfp_band\CSD_analysis\hit_go')
probe_list =  pd.read_csv('probe_id_csd.csv', header=None)

for session_id in session_list: 

    print(session_id)

    session = cache.get_ecephys_session(ecephys_session_id=session_id)

    stims = session.stimulus_presentations
    trials = session.trials
    all_start_times = stims.start_time

    # HIT GO TRIALS 
    hit_go_trials = stims[stims['is_change']==True]                             # all GO trials
    hit_go_trials = hit_go_trials[hit_go_trials['rewarded']==True]              # that were a hit
    hit_go_trials = hit_go_trials[hit_go_trials['stimulus_block']==0]           # during the ACTIVE period
    hit_go_trials = hit_go_trials[3:]                                           # the first three trials are autorewarded so reward = true even with no response + a larger volume that usual so ignore
    start_times = hit_go_trials['start_time'].values


    # CORRECT REJECT TRIALS
    # correct_reject_trials = trials [ trials ['correct_reject'] == True]                     # have to use the trials data because the stims data just gives whether sham_change = true but not whether it was correcty rejected. 
    # correct_reject_trials = correct_reject_trials['change_time_no_display_delay']   
    # correct_reject_start_times=[]
    # for st in correct_reject_trials.index:
    #     nearest = find_nearest(all_start_times, correct_reject_trials[st])
    #     n_df = np.float32([nearest])
    #     correct_reject_start_times= np.concatenate((correct_reject_start_times, n_df), axis=0)
    # start_times = correct_reject_start_times

    # PASSIVE START TIMES 
    # changed_images_passive = stims[ (stims['active']==False )  & 
    #                                         (stims['is_change']==True) ]
    # changed_images_passive_start_times = np.array(changed_images_passive.start_time)
    # start_times = changed_images_passive_start_times 


    # MISSED GO   
    # missed_go = stims[stims['is_change']==True]                             # all GO trials
    # missed_go = missed_go[missed_go['rewarded']==False]              # that were a hit
    # missed_go = missed_go[missed_go['stimulus_block']==0]           # during the ACTIVE period                                
    # missed_go = missed_go['start_time'].values
    # start_times =missed_go 

    idx = probe_list[probe_list[0] == session_id].index[0]  # find probe ID to orient to correct fldr. 

    probe_id = probe_list.iloc[idx,1]

    os.chdir(lfp_dir / f"{session_id}" /f"{probe_id}" )

    lfp_data = np.memmap(Path(r'lfp_band.dat'), dtype='int16', mode='r')
    lfp_data = np.reshape(lfp_data, [int(lfp_data.size/384), -1])               # CONVERT to volts later on when data extracted

    lfp_timestamps = np.load(Path(r'lfp_timestamps.npy'))
    lfp_timestamps = np.array (lfp_timestamps ) 

    with open("probe_info.json", "r") as f:
        probe_info = json.load(f)

    surface_channel =  probe_info["surface_channel"]  

    hz = 2500 

    print(lfp_timestamps.shape)
    print(lfp_data.shape) 
    
    # essentially just getting the part of the session that has the trials of interest, active / passive portion
    start_indx = np.argmin(np.abs(lfp_timestamps - start_times[0]-2))
    end_indx = np.argmin(np.abs(lfp_timestamps - (start_times[-1]+2) ))

    lfp_data = lfp_data [start_indx : end_indx , :] 
    lfp_timestamps = lfp_timestamps[start_indx : end_indx ] 


    
    # ----------------------------------------------------------
    # Go channel by channel to get them all on the same timestamp grid. 
    # --------

    ts  = lfp_timestamps

    # --- choose target sampling (regular grid) ---
    native_fs = 1.0 / np.median(np.diff(ts[ts >= 0]))
    target_fs =  2500   # or set explicitly, e.g. 2500.0
    dt = 1.0 / target_fs

    # Build a regular grid covering the valid time span
    t0 = ts[ts >= 0][0]
    t1 = ts[-1]
    t_grid = np.arange(t0, t1 + 0.5*dt, dt)  # +half-step so the end is included

    # --- interpolate channel-by-channel ---
    lfp_reg = np.empty((t_grid.size, lfp_data.shape[1]), dtype=np.float32)

    for ch in range(lfp_data.shape[1]):
        f = regular_grid_extractor_factory(ts, lfp_data, ch, method="linear")
        # RegularGridInterpolator in 1D is happiest with (..., 1) inputs:
        lfp_reg[:, ch] = f(t_grid.reshape(-1, 1))

   
    lfp_data = lfp_reg 
    print(lfp_data.shape) 
    lfp_reg = []
    lfp_timestamps = t_grid
    
    # ----------------------------------------------------------
    # Extract LFP around event windows 
    # ----------------------------------------------------------
     
    windows = []
    window_length = int((window_end_time-window_start_time) * hz)

    for stim_ts in start_times:
        # convert time to index
        start_idx = int( (stim_ts + window_start_time - lfp_timestamps[0]) * hz )
        end_idx = start_idx + window_length

        # bounds checking
        if start_idx < 0 or end_idx > lfp_data.shape[0]:
            continue
            
        windows.append(lfp_data[start_idx:end_idx,:])
        
    if len(windows) == 0:
        raise ValueError("There are no windows for these timestamps")

    windows = np.array(windows)
    print(windows.shape)


    # GET LFP AS trials x channels x time samples
    lfp = windows.transpose(0, 2, 1)
    print(lfp.shape)
    lfp = lfp * 0.000000195 * 0.5                                   # to get to volts, scaling factor + 0.5 gain stated in the json. 

    # ----------------------------------------------------------
    # Plot LFP to compare to Corbett's
    # ----------------------------------------------------------
    
    lfp_average_to_plot = np.average(lfp, axis=0)
    display_response_window(lfp_average_to_plot[::-1],
                        window_start_time,
                        window_end_time,
                        title="Average LFP Response To Stimulus, raw",
                        xlabel="Time relative to stimulus onset (s)",
                        ylabel="Approximate distance along probe (μm), \n[low values toward surface]",
                        cbar_label="Volts", vmin = -0.0004, vmax = 0.0005)



    # ----------------------------------------------------------
    # Clean, filter and plot LFP to compare to Corbett's again 
    # ----------------------------------------------------------

    # remove channels outside of the brain 
    reference_channels = list(range(383, surface_channel, -1)) 
    # also remove channel 191 (ref channel) and (subcortex, keep cortex only)
    reference_channels.extend(range(188, -1, -1))  # 191, 190, ..., 0   (includes 0)       

    reference_channels = [] 

    (cleaned_lfp, good_indices) = select_good_channels(lfp, reference_channels , 0.3) 

    filtered_lfp = filter_lfp_channels(cleaned_lfp,
                            hz, [1, 250], 3) 
    print(filtered_lfp.shape)


    def display_response_window(window, start_time, end_time, vmin=None, vmax=None, title="", xlabel="", ylabel="", cbar_label=""):
        fig, ax = plt.subplots(figsize=(6,6))

        img = ax.imshow(window, 
                        extent=[start_time, end_time, 384*10, 0], # probe is 3840 micrometers 
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax
                    ) 

        # make dotted line at stimulus onset
        ax.plot([0,0],[0, filtered_lfp.shape[0]*10], ':', color='white', linewidth=1.0)

        cbar = fig.colorbar(img, shrink=0.5)
        cbar.set_label(cbar_label)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


    lfp_average_to_plot = np.average(filtered_lfp, axis=0)
    display_response_window(lfp_average_to_plot[::-1],
                        window_start_time,
                        window_end_time,
                        title="Average LFP Response To Stimulus after cleaning and filtering",
                        xlabel="Time relative to stimulus onset (s)",
                        ylabel="Approximate distance along probe (μm), \n[low values toward surface]",
                        cbar_label="Volts", vmin = -0.00035, vmax = 0.00045)


    # # -------------------------------------------------------------------------------------
    # # Spatial interpolation to center align channels on the y axis - plot again to compare. 
    # # -------------------------------------------------------------------------------------

    actual_locs_all = make_actual_channel_locations(0, 384)    
    actual_locs     = actual_locs_all[good_indices]

    # interp_locs = make_interp_channel_locations(200, surface_channel) 
    interp_locs = make_interp_channel_locations(0, 384) 

    (interp_lfp , spacing) = interp_channel_locs(filtered_lfp,
                        actual_locs,
                        interp_locs,
                        'cubic') 


    lfp_average_to_plot = np.average(interp_lfp, axis=0)
    display_response_window(lfp_average_to_plot[::-1],
                        window_start_time,
                        window_end_time,
                        title="Average LFP Response To Stimulus after interpolation",
                        xlabel="Time relative to stimulus onset (s)",
                        ylabel="Approximate distance along probe (μm), \n[low values toward surface]",
                        cbar_label="Volts", vmin = -0.00035, vmax = 0.00045)

    # # # ----------------------------------------------------------
    # # # CSD and plot
    # # # ----------------------------------------------------------

    # Average the LFP   

    trial_mean_lfp = np.mean (interp_lfp,axis=0)
    print(trial_mean_lfp.shape)

    spacing = 0.02
    padded_lfp = np.pad(trial_mean_lfp,
                        pad_width=((1, 1), (0, 0)),
                        mode='edge')

    csd = (1 / (spacing ** 2)) * (padded_lfp[2:, :]
                                    - (2 * padded_lfp[1:-1, :])
                                    + padded_lfp[:-2, :])
    
    csd = csd * -0.3  ;                 # conductivity factor in extracellular tissue

    csd_channels = np.arange(0, trial_mean_lfp.shape[0])


    display_response_window(csd[::-1],
                        window_start_time,
                        window_end_time,
                        title="CSD",
                        xlabel="Time relative to stimulus onset (s)",
                        ylabel="Approximate distance along probe (μm), \n[low values toward surface]",
                        cbar_label="volts", vmin = -0.01, vmax = 0.01)


    csd_smoothed =  gaussian_filter(csd, sigma=(3,1))


    display_response_window(csd_smoothed[::-1],
                            window_start_time,
                            window_end_time,
                            title="Smoothed CSD",
                            xlabel="Time relative to stimulus onset (s)",
                            ylabel="Depth (µm)",
                            cbar_label="V/µm²", vmin = -0.005, vmax = 0.005)

    # THINGS TO SAVE 
    # THE RAW CSD 
    # THE AVERAGE LFP RESPONSE.

    os.chdir(dir_to_save) 

    # Save trial-mean LFP
    np.save(f"session_{session_id}_lfp.npy", trial_mean_lfp)

    # Save CSD
    np.save(f"session_{session_id}_csd.npy", csd)

