import os
import re
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import nrrd
import requests

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

from allensdk.core.reference_space_cache import ReferenceSpaceCache



class VBN_CCF:

    def __init__(self, manifest_path='manifest.json', resolution=10):
        reference_space_key = 'annotation/ccf_2017'
        self.resolution = resolution
        self.rspc = ReferenceSpaceCache(resolution,
                                        reference_space_key,
                                        manifest=manifest_path)
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
            streamlines_path = os.path.join(self._manifest_directory,
                                            'laplacian_10.nrrd')
            if os.path.exists(streamlines_path):
                self._streamlines, header = nrrd.read(streamlines_path)
            else:
                s = requests.get(
                    'https://www.dropbox.com/sh/7me5sdmyt5wcxwu/'
                    'AACFY9PQ6c79AiTsP8naYZUoa/laplacian_10.nrrd?dl=1'
                )
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
        volume_coords = [int(coord / self.resolution)
                         for coord in [ap_coord, dv_coord, lr_coord]]
        shape = self.annotation.shape
        if any([(v < 0 or v >= s) for v, s in zip(volume_coords, shape)]):
            print(f'Coordinate {[ap_coord, dv_coord, lr_coord]} is outside ccf')
            id_ = 0
        else:
            id_ = self.annotation[volume_coords[0],
                                  volume_coords[1],
                                  volume_coords[2]]
        return id_

    def get_structure_by_id(self, id_):
        if id_ == 0:
            structure = [{'name': 'outside_brain',
                          'acronym': 'outside_brain'}]
        else:
            structure = self.tree.get_structures_by_id([id_])
        return structure

    def get_structure_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        try:
            id_ = self.get_structure_id_by_coordinate(ap_coord,
                                                      dv_coord,
                                                      lr_coord)
            structure = self.get_structure_by_id(id_)
        except Exception as e:
            print(f'Could not get structure corresponding to id: {id_} '
                  f'due to error {e}')
            structure = [{}]
        return structure

    def get_structure_acronym_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        structure = self.get_structure_by_coordinate(ap_coord,
                                                     dv_coord,
                                                     lr_coord)[0]
        return structure.get('acronym', None)

    def get_cortical_layer_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        acronym = self.get_structure_acronym_by_coordinate(ap_coord,
                                                           dv_coord,
                                                           lr_coord)
        return self.get_cortical_layer_name_by_acronym(acronym)

    def get_cortical_layer_name_by_acronym(self, acronym):
        if acronym in ['CA1', 'CA2', 'CA3']:
            return ''

        try:
            first_num = re.findall(r'\d+', acronym)[0]
            first_num_ind = acronym.find(first_num)
            if first_num_ind < 0:
                return ''
            layer = acronym[first_num_ind:]
            return layer
        except IndexError:
            return ''

    def get_cortical_depth_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        volume_coords = [int(coord / 10)
                         for coord in [ap_coord, dv_coord, lr_coord]]
        try:
            cortical_depth = self.streamlines[volume_coords[0],
                                              volume_coords[1],
                                              volume_coords[2]]
        except IndexError as e:
            print(f'Coordinate {[ap_coord, dv_coord, lr_coord]} '
                  f'is outside of CCF')
            cortical_depth = 0
        return cortical_depth


c = VBN_CCF()


output_dir = r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Raw_Data'
lfp_dir = Path(r'\\psy-chaos\varelalab\Zoe\Allen_Institute_Project\Extracted_Data\lfp_band')

# Where to save the layer info npz + summary CSV
layer_info_dir = Path(r'Z:\Zoe\Allen_Institute_Project\Extracted_Data\lfp_band\layer_info')
layer_info_dir.mkdir(parents=True, exist_ok=True)

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
    cache_dir=Path(output_dir)
)

# Sessions to process
session_list = [
    1048196054, 1063010385, 1064639378, 1065908084,
    1067781390, 1081431006, 1086410738, 1091039376, 1092466205,
    1096935816, 1104297538, 1108528422, 1108531612, 1112515874,
    1115356973, 1121607504, 1130349290
]

# 1086410738 also removed as there is quoted to be no lfp available for that session. 
# REMOVED - RAW DATA NOT UPLOADED FOR 1120251466, 1122903357,


os.chdir(lfp_dir)
probe_list = pd.read_csv('probe_id_csd.csv', header=None)
# col 0: session_id, col 1: probe_id


session_layer_rows = []

for session_id in session_list:

    print(f"Processing session {session_id}")

    # Find probe_id for this session
    idx = probe_list[probe_list[0] == session_id].index[0]
    probe_id = int(probe_list.iloc[idx, 1])

    # Read surface channel for this probe
    probe_dir = lfp_dir / f"{session_id}" / f"{probe_id}"
    with open(probe_dir / "probe_info.json", "r") as f:
        probe_info = json.load(f)
    surface_channel = int(probe_info["surface_channel"])

    # Get session object and units/channels
    ephys_session = cache.get_ecephys_session(ecephys_session_id=session_id)

    units = ephys_session.get_units(amplitude_cutoff_maximum=np.inf,
                                    presence_ratio_minimum=-np.inf)
    channels = ephys_session.get_channels()

    probe_channels = channels[
    channels['probe_id'] == probe_id
    ].copy()

    probe_channels = probe_channels[
    probe_channels['structure_acronym'] == 'VISp'
    ].copy()

    probe_channels['cortical_layer'] = probe_channels.apply(
    lambda row: c.get_cortical_layer_by_coordinate(
        row['anterior_posterior_ccf_coordinate'],
        row['dorsal_ventral_ccf_coordinate'],
        row['left_right_ccf_coordinate']
    ),
    axis=1
    )
    
    #----------------------
    #   Get the minimum and maximum DV channel number for channels in each layer
    #----------------

    # Layer 1
    l1 = probe_channels[probe_channels['cortical_layer'] == '1']
    l1_d = int ( max (l1['probe_channel_number']) ) 
    l1_v = int ( min (l1['probe_channel_number']) ) 

    # layer 23
    l23 = probe_channels[probe_channels['cortical_layer'].isin(['2', '3', '2/3', '23'])] 
    l23_d = int ( max (l23['probe_channel_number']) ) 
    l23_v = int ( min (l23['probe_channel_number'])  )   

    # Layer 4
    l4 = probe_channels[probe_channels['cortical_layer'] == '4']
    l4_d = int ( max (l4['probe_channel_number']) ) 
    l4_v = int ( min (l4['probe_channel_number']) ) 

    # Layer 5
    l5 = probe_channels[probe_channels['cortical_layer'] == '5']
    l5_d = int ( max (l5['probe_channel_number']) ) 
    l5_v = int ( min (l5['probe_channel_number']) ) 

    # Layer 6 (6a and 6b)
    l6 = probe_channels[probe_channels['cortical_layer'].isin(['6a', '6b'])]
    l6_d = int ( max (l6['probe_channel_number']) ) 
    l6_v =int ( min (l6['probe_channel_number']) ) 
 

    #---------------------
    #   L4/5 anchor (for CSD averaging in next script. 
    # ----------------------

    anchor_L45 = np.mean([l4_v, l5_d])


    #---------------
    #   Create summary csv of this info 
    #------------------------

    session_layer_rows.append(
        (session_id, probe_id, surface_channel, anchor_L45,
         l1_d, l1_v, l23_d, l23_v, 
         l4_d, l4_v, l5_d, l5_v, l6_d, l6_v ) ) 

#--------------
#   Save summary CSV across sessions
#-------------------

df_layers = pd.DataFrame(
    session_layer_rows,
    columns=[
        'session_id', 'probe_id', 'surface_channel', 'anchor_L45',
        'l1_d', 'l1_v', 'l23_d', 'l23_v', 
        'l4_d','l4_v', 'l5_d', 'l5_v', 'l6_d', 'l6_v'
    ]
)

csv_path = layer_info_dir / 'layer_channel_summary.csv'
df_layers.to_csv(csv_path, index=False)
print(f"\nSaved summary CSV: {csv_path}")
