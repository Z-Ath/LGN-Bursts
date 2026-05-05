# LGN-Bursts
Code used to create figures in:

Thalamic Burst Firing Encodes Task-specific Content During Visual Deviance  
Zoe Atherton, Bolin Shen, Logan Becker, Yushun Dong, Carmen Varela  
Florida State University

Code used to:  
1) Extract trials and sessions using the Allen-SDK
2) Extract burst spikes and firing rates (Figure 1)
3) Decode GO trial outcome and image ID (Figure 2 a-b)
4) Create image embeddings and train and test XGBoost models (Figure2 d-g)
5) Complete CSD analysis and burst rate correlations with sinks (Figure 2h-l)

More details: 

5) CSD analysis
completed according to 'https://alleninstitute.github.io/openscope_databook/first-order/current_source_density.html
Location of layers and the layer V/VI boundary was calculated based on the units within each layer and extracting the location of the most dorsal and ventral unit. - this informed layer boundaries for burst correlation analysis and point of reference to average CSDs across sessions. 
