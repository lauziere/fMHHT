# Fast Multiple Hypothesis Hypergraph Tracking (fMHHT) for Posture Tracking in Embryonic *C. elegans*

This repository contains a Python3 implementation of Fast Multiple Hypothesis Hypergraph Tracking (fMHHT) with application to Posture Tracking in Embryonic *C. elegans* as described in:

- Andrew Lauziere, Evan Ardiel, Stephen Xu, Hari Shroff. [*Multiple Hypothesis Hypergraph Tracking for Posture Identification in Embryonic Caenorhabditis elegans*](https://arxiv.org/abs/2111.06425)

## Methods

MHHT adapts hypothesis oriented multiple hypothesis tracking (MHT) to further evaluate sampled hypotheses via hypergraphical models. The data association step is enhanced to more effectively disambiguate competeting hypotheses than the traditional MHT paradigm. The method is only effective in presence of true object interdepence; i.e. when object behaviors are contingent on one another. Annotated data can be used to estimate parameters of graphical models, enhancing the accuracy of the robust data association step. 

This code uses the optimized Murty's algorithm from Miller, Stone, and Cox. The suggested changes in Cox and Miller to make Murty's algorithm more suited to the data assoication problem are also implemented. 

In particular, biologically guided graphical models are used to perform seam cell identification in embryonic *C. elegans*. The 20 or 22 seam cells together are used to approximate the posture of the coiled embryo. Volumetric images of the developing embryo are captured throughout late-stage embryogensis. A convolutional neural network detects seam cell nuclei while MHHT is used to perform MOT on the seam cell nuclei. 
 
## Installation

The code requires Python 3.6 or higher, and requires NumPy>=1.19.5, Pandas>=1.1.5, and Scipy>=1.5.4. Cloning the repository and installing the requirements in a new Python 3.6.5 envirionment is advised.  
  
## Configuration

### Dataset: 'Embryo1' or 'Embryo2'

Two imaged embryos are annotated for this research. The algorithm can be run on either dataset. 

### Detection method: 'Annotations', '3D-UNet', or 'IFT-Watershed'

A 3D U-Net is trained to perform semantic segmentation on the image volumes. A thresholding on network output followed by connected components analysis yields the final detection set ('3D-UNet'). A more traditional Watershed algorithm for blob detection is also applied to the image volumes ('IFT-Watershed'). MHHT can also be applied on the annotatied data as well, simulating the scenario in which a detection algorithm perfectly locates seam cell nuclei ('Annotations'). 

### Interpolation method: 'Last' or 'Graph'

The adjustments to traditional MHT allow for missed detection interpolation. 'Graph' will used the provided graphical model, while 'Last' uses the position of the cell in the prior frame. 

### Interpolation cost. Default is the gate size d.

An interpolation tax is built in to dissuade spurious interpolations, forcing MHHT to use detections. Lower values will favor interpolations near detections to better estimate embryonic posture, while larger values will bias MHHT away from point interpolation. 

### Gate: d (microns): 

The gate *d* is a measurement used in the linear program for matching a missing track dummy to an existing track. The application is in microns (μm) while the linear program follows the global nearest neighbor (GNN). Thus, *d* in this case is a micron distance from a track in which detections may be associated. 

### Model: 'GNN' (MHT w/ K=1 & N=1), 'MHT', 'Embryo', 'Movement', 'Posture', or 'Posture-Movement'

Unary models 'GNN' and 'MHT' are able to be performed in the MHHT framework. The unparameterized graphical model *Embryo* follows from the proposed embryo graph ( see Figure 3 on ArXiv). Data driven models *Movement*, *Posture*, and the hybrid model: *Posture-Movement* leverage the embryo graph. The hybrid model, *Posture-Movement* yields the strongest results. 

### Search width K:

*K* specifies the number of best solutions of the GNN explored at each scan. Larger *K* will increase runtime (*K^N*), but allow for the exploration of potentially lower cost hypotheses. 

### Search depth N:

The tree depth *N* defines how many future frames are considered to yield tracks at the current frame. *N*=1 is a standard frame-to-frame tracking approach. Using future frames improves performance but causes exponential increases in computation. 


|         | Pre-Q           | Post-Q  |
| ------------- |:-------------:| -----:|
| Embryo1     | 2415-35099 | 35100-53968 |
| Embryo2      | 0-29660	      |   N/A |

Annotations for Embryo1 begin at image number 2415, while annotations begin immediately for Embryo2. However, the *Q* cell split (20 to 22 cells) is not recorded in Embryo2. StartFrame and EndFrame must be within the annotated ranges above. 

### Notebook tracking cost threshold for plotting: cost_threshold & print_interval

The semi-automated tracking tool will display the tracked embryo at the previous frame (red) and current predictions (blue) when the total tracking cost exceeds cost_threshold or at each print_interval value. The semi-automated tool is used to ensure the integrity of the embryo is being maintained throughout tracking. 

### Use

The IPython notebook tracking.pynb will take the configuration as input to perform semi-automated tracking. At each step output is saved in CSV form. When a cost exceeds cost_threshold *or* the frame number is on a print_interval, the plot is rendered for inspection. If it is correct, the user can skip the step and continue. However, if the tracks are incorrect, the CSV can be altered and placed in the *corrections* folder. It will then be read in when the user enters 'done', and the tracks will be overwritten with the user supplied coordinates. 

## References

\[1\] Murty, K. (1968). An Algorithm for Ranking all the Assignments in
Order of Increasing Cost. *Operations Research, 16*(3), 682-687.
Retrieved from <http://www.jstor.org/stable/168595>

\[2\] https://github.com/arg0naut91/muRty

\[3\] Seyed Hamid Rezatofighi, Anton Milan, Zhen Zhang, Qinfeng Shi, Anthony Dick, and Ian Reid. Joint Proba- bilistic Data Association Revisited. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 3047–3055, Santiago, Chile, December 2015. IEEE. ISBN 978-1-4673-8391-2. doi: 10.1109/ICCV.2015.349. URL http://ieeexplore.ieee.org/document/7410706/.

\[4\] http://www.milanton.de/#publications

\[5\] Khuloud Jaqaman, Dinah Loerke, Marcel Mettlen, Hirotaka Kuwata, Sergio Grinstein, Sandra L. Schmid, and Gaudenz Danuser. Robust single-particle tracking in live-cell time-lapse sequences. Nature Methods, 5(8):695– 702, August 2008. ISSN 1548-7105. doi: 10.1038/nmeth.1237. URL https://www.nature.com/articles/ nmeth.1237. Number: 8 Publisher: Nature Publishing Group.


