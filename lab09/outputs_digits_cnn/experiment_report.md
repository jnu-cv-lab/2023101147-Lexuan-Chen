# CNN Optimizer And Learning Rate Experiment

## Task 1: CNN Model
Used a two-layer CNN: Conv2d(1->16), MaxPool, Conv2d(16->32), MaxPool, FC(64), FC(10).

## Task 2: Optimizer Comparison
| Optimizer | LR | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 0.01 | 2.2880 | 2.2839 | 0.1599 | 0.1704 | 0.1556 |
| SGD+Momentum | 0.01 | 2.1122 | 2.0093 | 0.5139 | 0.4593 | 0.5519 |
| Adam | 0.01 | 0.1755 | 0.2901 | 0.9467 | 0.9296 | 0.9222 |

## Task 3: Learning Rate Comparison
| Optimizer | LR | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| Adam | 0.1 | 1.1805 | 1.6910 | 0.5990 | 0.5667 | 0.6259 |
| Adam | 0.01 | 0.1755 | 0.2901 | 0.9467 | 0.9296 | 0.9222 |
| Adam | 0.001 | 1.2112 | 0.9627 | 0.7741 | 0.7630 | 0.7889 |

## Task 4: Convolution Kernel Visualization
Saved first-layer convolution kernels to `conv_kernels.png`.
Kernels with positive/negative color changes often respond to local edge, stroke direction, and brightness contrast.

## Task 5: Feature Map Visualization
Saved first-layer feature maps to `feature_maps.png`.
Different maps emphasize different digit strokes, corners, edge positions, and local texture responses.

## Task 6: Misclassified Samples
Saved misclassified examples to `misclassified_samples.png`.
Observed wrong labels: true 9/pred 8, true 9/pred 8, true 8/pred 1, true 3/pred 8, true 8/pred 1, true 9/pred 8, true 8/pred 1, true 8/pred 1.
Errors usually happen when different digits share similar strokes, when writing is ambiguous, or when the sample is very sparse.

## Task 7: Confusion Matrix
Saved confusion matrix to `confusion_matrix.png`.
Most visible confusion count: 6 for 8->1.

Best model selected for visual analysis: Adam.