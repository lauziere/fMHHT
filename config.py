
import os

config = {}

# Dataset: 'Embryo1' or 'Embryo2'
Dataset = 'Embryo1'

# Detection method: 'Annotations', '3D-UNet', or 'IFT-Watershed'
Detection = '3D-UNet'

# Interpolation method: 'Last' or 'Graph'
Interpolation = 'Graph'

# Interpolation cost. Default is the gate size d.
Interpolation_cost = 5.0

# Gate: d (microns): 
d = 5.0

# Model: 'GNN' (MHT w/ K=1 & N=1), 'MHT', 'Embryo', 'Movement', 'Posture', or 'Posture-Movement'
Model = 'Posture-Movement'

# Search width K:
K = 15

# Search depth N:
N = 3

# Start frame & End frame
#		  	pre-Q			post-Q
# Embryo1	2415-35099		35100-53968			
# Embryo2	0-29660			N/A
StartFrame = 3465
EndFrame = 4475

# Notebook tracking cost threshold for plotting 
cost_threshold = 0
print_interval = 25

# Q is determined by Dataset and StartFrame:
Q = 1 if ((Dataset=='Embryo1') and (StartFrame >= 35100)) else 0

# Build paths:
Home = os.getcwd()
annotation_path = os.path.join(Home, 'Data', Dataset, 'Annotations.npy')
prediction_path = os.path.join(Home, 'Data', Dataset, Detection + '.npy')
posture_weights_path = os.path.join(Home, 'Estimates', 'Embryo1' if Dataset=='Embryo2' else 'Embryo2', 'Posture_' + str(Q) + '.npy')
movement_weights_path = os.path.join(Home, 'Estimates', 'Embryo1' if Dataset=='Embryo2' else 'Embryo2', 'Movement_' + str(Q) + '.npy')

# Load configuration
config['Dataset'] = Dataset
config['Q'] = Q
config['Detection'] = Detection
config['Interpolation'] = Interpolation
config['Interpolation_cost'] = Interpolation_cost
config['d'] = d if Detection != 'Annotations' else 1e6
config['Model'] = Model
config['K'] = K if Model != 'GNN' else 1
config['N'] = N if Model != 'GNN' else 1
config['StartFrame'] = StartFrame
config['EndFrame'] = EndFrame
config['InitialFrame'] = 2415 if Dataset == 'Embryo1' else 0
config['Annotation_path'] = annotation_path
config['Prediction_path'] = prediction_path
config['Posture_weights_path'] = posture_weights_path
config['Movement_weights_path'] = movement_weights_path
config['Home'] = Home
config['Cost_threshold'] = cost_threshold
config['Print_interval'] = print_interval

if ((StartFrame < 2415) and (Dataset == 'Embryo1')):
	raise Exception('Embryo1 begins at frame 2415. Set StartFrame >= 2415.')
if EndFrame <= StartFrame:
	raise Exception('Pick a valid StartFrame and EndFrame. StartFrame:', StartFrame, 'EndFrame:', EndFrame)
