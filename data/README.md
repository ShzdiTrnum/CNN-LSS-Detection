# data/

Download the Lumbar Spine MRI Dataset from Mendeley Data:
  https://data.mendeley.com/datasets/k57fr854j2/2

  Sudirman, S. et al. (2019). Lumbar Spine MRI Dataset. Mendeley Data, V2.

Place the downloaded .npz file here and update DATA_PATH in src/data_utils.py:
  DATA_PATH = 'data/Orignal_Data.npz'

Dataset details:
  515 patients, 1545 axial-view MRI images
  Resolution: 320x320 px (cropped to 240x240 for training)
  Classes: Normal (0), Mild (1), Moderate (2), Severe (3)
  Regions: IVD, PE, TS, AAP
