### WIDER FACE val: official AP

| config | easy | medium | hard | n_images | n_errors |
|---|---|---|---|---|---|
| scrfd__full640 | 0.950 | 0.936 | 0.824 | 3226 | 0 |
| scrfd__full1280 | 0.920 | 0.925 | 0.892 | 3226 | 0 |
| scrfd__tiled | 0.933 | 0.931 | 0.886 | 3226 | 0 |
| retinaface__full640 | 0.944 | 0.910 | 0.665 | 3226 | 0 |
| retinaface__full1280 | 0.955 | 0.943 | 0.872 | 3226 | 0 |
| retinaface__tiled | 0.943 | 0.932 | 0.840 | 3226 | 0 |

### WIDER FACE val: recall by face height, score >= 0.5

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/image |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | 0.139 | 0.556 | 0.754 | 0.851 | 0.911 | 0.962 | 0.750 |
| retinaface__full640 | 0.000 | 0.036 | 0.358 | 0.641 | 0.834 | 0.947 | 0.377 |
| retinaface__tiled | 0.054 | 0.448 | 0.713 | 0.821 | 0.905 | 0.958 | 0.824 |
| scrfd__full1280 | 0.202 | 0.554 | 0.727 | 0.820 | 0.876 | 0.903 | 0.307 |
| scrfd__full640 | 0.015 | 0.199 | 0.556 | 0.747 | 0.864 | 0.944 | 0.203 |
| scrfd__tiled | 0.156 | 0.524 | 0.723 | 0.821 | 0.881 | 0.923 | 0.341 |

### WIDER FACE val: recall by face height, score >= 0.3

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/image |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | 0.210 | 0.642 | 0.802 | 0.880 | 0.925 | 0.967 | 1.416 |
| retinaface__full640 | 0.000 | 0.077 | 0.448 | 0.703 | 0.861 | 0.956 | 0.791 |
| retinaface__tiled | 0.104 | 0.549 | 0.768 | 0.858 | 0.923 | 0.965 | 1.564 |
| scrfd__full1280 | 0.467 | 0.757 | 0.841 | 0.897 | 0.924 | 0.931 | 1.891 |
| scrfd__full640 | 0.087 | 0.489 | 0.753 | 0.857 | 0.915 | 0.964 | 1.444 |
| scrfd__tiled | 0.395 | 0.740 | 0.841 | 0.897 | 0.925 | 0.951 | 2.131 |

### ChokePoint sample: recall by face height, score >= 0.5

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/frame |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | nan | nan | nan | nan | 1.000 | 0.981 | 0.115 |
| retinaface__full640 | nan | nan | nan | nan | 1.000 | 0.988 | 0.094 |
| retinaface__tiled | nan | nan | nan | nan | 0.997 | 0.973 | 0.140 |
| scrfd__full1280 | nan | nan | nan | nan | 1.000 | 0.978 | 0.102 |
| scrfd__full640 | nan | nan | nan | nan | 1.000 | 0.979 | 0.093 |
| scrfd__tiled | nan | nan | nan | nan | 1.000 | 0.984 | 0.097 |

### ChokePoint sample: recall by face height, score >= 0.3

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/frame |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | nan | nan | nan | nan | 1.000 | 0.981 | 0.130 |
| retinaface__full640 | nan | nan | nan | nan | 1.000 | 0.988 | 0.099 |
| retinaface__tiled | nan | nan | nan | nan | 0.997 | 0.973 | 0.171 |
| scrfd__full1280 | nan | nan | nan | nan | 1.000 | 0.978 | 0.133 |
| scrfd__full640 | nan | nan | nan | nan | 1.000 | 0.979 | 0.111 |
| scrfd__tiled | nan | nan | nan | nan | 1.000 | 0.984 | 0.135 |

### TinyFace / SurvFace crops: detection rate by native crop size (longest side, px)

| detector | dataset | all | <16 | 16-24 | 24-32 | 32-48 | >48 |
|---|---|---|---|---|---|---|---|
| scrfd | tinyface | 0.561 | 0.100 | 0.381 | 0.563 | 0.909 | 0.522 |
| scrfd | survface | 0.425 | 0.256 | 0.326 | 0.496 | 0.579 | 0.715 |
| retinaface | tinyface | 0.589 | 0.138 | 0.421 | 0.586 | 0.930 | 0.413 |
| retinaface | survface | 0.435 | 0.308 | 0.389 | 0.486 | 0.531 | 0.676 |

### Runtime (median ms per 720p frame, batch 1)

| config | median_ms | p95_ms | n |
|---|---|---|---|
| scrfd__full640 | 24.787 | 26.648 | 100 |
| scrfd__full1280 | 88.296 | 93.345 | 100 |
| scrfd__tiled | 96.015 | 108.602 | 100 |
| retinaface__full640 | 30.394 | 31.226 | 100 |
| retinaface__full1280 | 105.764 | 109.888 | 100 |
| retinaface__tiled | 122.075 | 126.945 | 100 |

### Selected detector (Phase 6)

```

{
 "config": "scrfd__full1280",
 "recall": 0.7629861421546714,
 "median_ms": 88.29566050007998,
 "candidates": [
  {
   "config": "scrfd__full640",
   "recall": 0.629682610639249,
   "median_ms": 24.78697549986464
  },
  {
   "config": "scrfd__full1280",
   "recall": 0.7629861421546714,
   "median_ms": 88.29566050007998
  },
  {
   "config": "scrfd__tiled",
   "recall": 0.7606616003576219,
   "median_ms": 96.01534500052367
  },
  {
   "config": "retinaface__full640",
   "recall": 0.467411712114439,
   "median_ms": 30.3943909998452
  },
  {
   "config": "retinaface__full1280",
   "recall": 0.7916852928028609,
   "median_ms": 105.7636294999611
  },
  {
   "config": "retinaface__tiled",
   "recall": 0.7546714349575324,
   "median_ms": 122.07454650024374
  }
 ],
 "rule": {
  "score": 0.5,
  "bins": [
   "16-24",
   "24-32"
  ],
  "max_ms": 100,
  "tie_recall": 0.01
 }
}

```