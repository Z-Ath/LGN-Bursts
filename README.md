# LGN-Bursts
Code used to create figures in:

Thalamic Burst Firing Encodes Task-specific Content During Visual Deviance  
Zoe Atherton, Bolin Shen, Logan Becker, Yushun Dong, Carmen Varela  
Florida State University

Code used to:  
1) Extract trials and sessions using the Allen-SDK
2) Decode GO trial outcome and image ID using Support Machine Vector Classifiers (SVCs) (Figure 2 a-b)
3) Create image embeddings and train and test XGBoost models (Figure2 d-g)
4) Complete CSD analysis and burst rate correlations with sinks (Figure 2h-l)

More details: 

**2) Decoding with SVC's**   
Population decoding was performed using a linear Support Vector Classifier (SVC27, implemented with LinearSVC (scikit-learn).  
Classifier performance was evaluated using the area under the receiver operating characteristic curve (ROC-AUC   

**3) Machine learning approach to predict single neuron firing rates** 

**4) CSD analysis**  
4A) CSD_calculation - completed according to 'https://alleninstitute.github.io/openscope_databook/first-order/current_source_density.htmL'  
4B) Channel_layers - assign cortical layers to each probe channel to get the layer IV/V boundary for CSD averaging across sessions + boundaries for layer specific sinks analysis.  
4C) CSD sink correlations with Burst (/Tonic) firing. 
