# Architecture Notes

## Design choices

### Filter progression: 32 -> 64 -> 128 -> 256 -> 512
Each conv block doubles the number of feature maps while halving spatial
dimensions via MaxPool(2,2). After 5 blocks: 240/32 = 7.5 -> effective
spatial size ~7x7 before flatten.

### Dropout rate 0.14
Relatively low dropout (vs typical 0.5) prevents aggressive regularisation
on a small medical imaging dataset, allowing the model to retain learned
features. Applied uniformly after every Conv block and FC layer.

### Single-ROI > Multi-ROI
AAP-only (Single-ROI) consistently outperforms all four regions (Multi-ROI)
because the AAP is the primary diagnostic region for foraminal stenosis.
Multi-ROI introduces irrelevant texture from IVD/PE/TS that adds noise.

### 12.5k dataset optimal
With 5k the model shows underfitting (validation accuracy oscillates).
With 10k there is slight underfitting on multi-ROI.
With 12.5k both datasets train to convergence within 34 epochs.

## Legacy Keras imports (corrected in this repo)

Original notebook used deprecated:
  from keras.layers.normalization import BatchNormalization
  from keras.layers.advanced_activations import LeakyReLU

These are not used in the final architecture but were imported.
This repo uses TF2-compatible imports from tensorflow.keras.
