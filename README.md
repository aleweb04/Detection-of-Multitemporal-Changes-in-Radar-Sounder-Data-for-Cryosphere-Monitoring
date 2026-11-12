# Detection-of-Multitemporal-Changes-in-Radar-Sounder-Data-for-Cryosphere-Monitoring


## Abstract

The work focused on developing an automatic method for detecting multitemporal changes in Antarctic ice-sheet structures using radar sounder data. The main motivation was to study the automatic, via neural network, monitoring methods of the cryosphere and making it possible to identify changes that are related to climate change and changes on the underground structure of the criosphere.


## Train
A neural network was trained using 2014 radargram patches in an unsupervised autoencoder configuration, learning to reconstruct the input images. The trained network was then applied to data from both 2014 and 2016 to extract meaningful features while preserving spatial information and reducing noise.

## TV Loss

During network training, TV Loss was introduced to encourage spatial continuity in the extracted feature maps. It penalizes abrupt differences between neighbouring pixels, helping the network produce smoother and more coherent regions.

The comparison between models trained with and without TV Loss showed that its inclusion led to more compact and continuous change clusters, reducing scattered detections and improving the definition of horizontal stratigraphic changes.

## Network architecture (directory models and results)

Different configurations are compared:
- Shallow-CNN only.
- Shallow-CNN + ASPP.
- Shallow-CNN + ASPP + BiConvLSTM.
- Training with MSE only or with MSE and TV Loss.



## Feature extraction and selection

The proposed method uses the extracted features from the different models and selects them via different methods: variance and SSIM. This method allows the change detection on a global and larger scale because we can modify the neural networks as we need to improve the change detection on the stratigrafy of the criosphere.

## Variance-Based Feature Selection

Variance was also used to select the most informative feature maps. It measures how much the values within a feature map differ from their mean.

Feature maps with higher variance were selected because they showed greater variation in their pixel values and were considered more informative for detecting changes.

## SSIM (Structure Similarity Index Measure)

SSIM was used as a feature selection method to compare the structural similarity between feature maps. It considers three aspects: luminance, contrast and spatial structure.

The similarity values were calculated between feature maps, and those with lower SSIM values were selected because they represented more distinct features. This helped identify complementary information and highlight different types of changes.

The results showed that SSIM improved the selection of feature maps, producing cleaner change detection overlays, with fewer isolated pixels and more concentrated regions of change.

## Change vector analysis

The extracted features were compared and analysed using different change detection approaches, mainly Change Vector Analysis (CVA) and Sequential Spatial Change Vector Analysis (S2CVA). Feature selection and clustering techniques were also applied to identify significant differences and group similar changes.

## Change maps

Finally, the detected changes were mapped onto the original radargrams to provide a spatial representation of their location, extent and different patterns.
