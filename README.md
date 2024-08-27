# Anatomical Plausibility Validation for Cardiac Ultrasound Segmentation

## Overview

A significant challenge in AI-based segmentation of anatomical structures is ensuring anatomical plausibility, especially as image quality impacts AI model performance. The goal of this project was to develop an algorithm to classify and verify the anatomical plausibility of AI-generated segmentations of cardiac structures in the apical two-chamber view.

## Objectives

- **Ensure anatomical plausibility:** Address the challenge of anatomical implausibility in AI-generated segmentations of cardiac structures by developing an algorithm to classify and verify plausibility.

## Tasks

### 1. Detection of Holes within Anatomical Structures (Criterion 1 & 2)
- Developed an algorithm to detect holes within various anatomical structures.
- The algorithm identifies the specific structure containing the hole and detects intersections between two structures.

### 2. Detection of Disconnected Components (Criterion 3)
- Created a method to identify disconnected components within different structures.

### 3. Detection of Incorrect Contacts between Structures (Criterion 4)
- Developed an algorithm to detect incorrect contacts between various structures, ensuring:
  - Myocardium does not contact the atrium.
  - Endocardium does not contact the background.

### 4. Myocardium Thickness Calculation (Criterion 5 & 6)
- Calculated the thickness of the myocardium, overcoming challenges presented by the myocardial bases.
- Subtasks included:
  - Rotating the myocardium in a consistent direction using the principal components of the myocardium-endocardium intersection.
  - Trimming the myocardium to remove base regions.
  - Calculating the ratio between minimum and maximum thickness.
  - Determining the ratio between average thickness and endocardium width.

### 5. Curvature Analysis (Criterion 7)
- Introduced a new criterion for curvature analysis, not previously covered in literature.
- Steps involved:
  - Obtaining and ordering the contours of various structures.
  - Interpolating contours to ensure uniform point spacing.
  - Calculating the second derivative at each contour point (using a 30-point window) to identify anomalous curvatures.

### 6. Algorithm Testing and Validation
- Conducted tests using simulated segmentations with known anatomical implausibilities (holes, disconnected components, etc.).
- Defined threshold values for various criteria by running the algorithm on manually segmented, anatomically plausible images.
- Validated the algorithm by applying it to segmentations predicted by an AI model, confirming its effectiveness in identifying plausible segmentations.

## Conclusion

The developed algorithm successfully classifies and verifies the anatomical plausibility of AI-generated cardiac structure segmentations. The project achieved its goal of improving the reliability of AI models in clinical settings by ensuring that segmentations are anatomically plausible, thereby enhancing the overall quality of cardiac ultrasound diagnostics.

## Future Work

- Further refinement of the algorithm based on a broader set of validation cases.
- Integration with AI models for real-time segmentation plausibility checks.
- Expansion of the algorithm to include additional anatomical views and structures.

## Acknowledgements

Special thanks to [INESCTEC/CBER] for the support and resources provided throughout this project.

