# ROB7 group 760 Advanced Robotic Perception Miniproject

This repository contains the code used for the miniproject.
In order to replicate our results, download the dataset from:
http://pascal.inrialpes.fr/data/human/
and place it along with the other files. The scripts assume the dataset folder to be in the CWD

6 python scripts have been made:

**annotation_parser.py:**
- script to parse annotations in the images, as well as handling paths etc.
- This should not be run on its own

**calc_hog.py:**
- Implementation of HOG features
- This should not be run on its own

**feature_extractor.py:**
- Script which loads all images from the dataset and calculates HOG features for each detection window.
- The resulting samples are saved as pickle files, such that it is only necessary to do this once

**training.py:**
- Loads dataset samples created by _feature_extractor.py_ and trains the SVM.
- The trained classifier is also dumped to a pickle.

**demo.py:**
- Demo script which uses the saved classifier and evaluates it on the test set, as well as applying the trained classifier on test images using a sliding window approach with non maximal suppression

**non_maximal_suppression.py:**
- Simple of non maximal suppression, using confidence if available, otherwise area to decide which box to keep. (based on an overlap threshold)

