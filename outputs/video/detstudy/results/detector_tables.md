### WIDER FACE val: official AP

| config | easy | medium | hard | n_images | n_errors |
|---|---|---|---|---|---|
| scrfd__full640 | 0.948 | 0.935 | 0.813 | 3226 | 0 |
| scrfd__full1280 | 0.918 | 0.923 | 0.888 | 3226 | 0 |
| scrfd__tiled | 0.933 | 0.931 | 0.883 | 3226 | 0 |
| retinaface__full640 | 0.944 | 0.910 | 0.665 | 3226 | 0 |
| retinaface__full1280 | 0.955 | 0.943 | 0.872 | 3226 | 0 |
| retinaface__tiled | 0.943 | 0.932 | 0.840 | 3226 | 0 |

### WIDER FACE val: recall by face height, score >= 0.5

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/image |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | 0.139 | 0.556 | 0.754 | 0.851 | 0.911 | 0.962 | 0.750 |
| retinaface__full640 | 0.000 | 0.036 | 0.358 | 0.641 | 0.834 | 0.947 | 0.377 |
| retinaface__tiled | 0.054 | 0.448 | 0.713 | 0.821 | 0.905 | 0.958 | 0.824 |
| scrfd__full1280 | 0.196 | 0.551 | 0.725 | 0.819 | 0.876 | 0.903 | 0.325 |
| scrfd__full640 | 0.013 | 0.192 | 0.552 | 0.743 | 0.863 | 0.944 | 0.230 |
| scrfd__tiled | 0.147 | 0.521 | 0.724 | 0.821 | 0.881 | 0.924 | 0.362 |

### WIDER FACE val: recall by face height, score >= 0.3

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/image |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | 0.210 | 0.642 | 0.802 | 0.880 | 0.925 | 0.967 | 1.416 |
| retinaface__full640 | 0.000 | 0.077 | 0.448 | 0.703 | 0.861 | 0.956 | 0.791 |
| retinaface__tiled | 0.104 | 0.549 | 0.768 | 0.858 | 0.923 | 0.965 | 1.564 |
| scrfd__full1280 | 0.453 | 0.749 | 0.839 | 0.894 | 0.922 | 0.931 | 1.945 |
| scrfd__full640 | 0.075 | 0.467 | 0.745 | 0.850 | 0.915 | 0.964 | 1.507 |
| scrfd__tiled | 0.370 | 0.733 | 0.839 | 0.895 | 0.924 | 0.953 | 2.205 |

### ChokePoint sample: recall by face height, score >= 0.5

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/frame |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | nan | nan | nan | nan | 1.000 | 0.981 | 0.115 |
| retinaface__full640 | nan | nan | nan | nan | 1.000 | 0.988 | 0.094 |
| retinaface__tiled | nan | nan | nan | nan | 0.997 | 0.973 | 0.140 |
| scrfd__full1280 | nan | nan | nan | nan | 1.000 | 0.978 | 0.102 |
| scrfd__full640 | nan | nan | nan | nan | 1.000 | 0.981 | 0.092 |
| scrfd__tiled | nan | nan | nan | nan | 1.000 | 0.984 | 0.097 |

### ChokePoint sample: recall by face height, score >= 0.3

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/frame |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | nan | nan | nan | nan | 1.000 | 0.981 | 0.130 |
| retinaface__full640 | nan | nan | nan | nan | 1.000 | 0.988 | 0.099 |
| retinaface__tiled | nan | nan | nan | nan | 0.997 | 0.973 | 0.171 |
| scrfd__full1280 | nan | nan | nan | nan | 1.000 | 0.978 | 0.132 |
| scrfd__full640 | nan | nan | nan | nan | 1.000 | 0.981 | 0.110 |
| scrfd__tiled | nan | nan | nan | nan | 1.000 | 0.984 | 0.132 |

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
| scrfd__full640 | 20.540 | 23.046 | 100 |
| scrfd__full1280 | 88.366 | 90.408 | 100 |
| scrfd__tiled | 91.106 | 95.175 | 100 |
| retinaface__full640 | 30.869 | 31.774 | 100 |
| retinaface__full1280 | 107.726 | 109.642 | 100 |
| retinaface__tiled | 124.439 | 126.737 | 100 |

### Selected detector (Phase 6)

```

{
 "config": "scrfd__full1280",
 "recall": 0.7611086276262852,
 "median_ms": 88.36647399994035,
 "candidates": [
  {
   "config": "scrfd__full640",
   "recall": 0.6256593652212785,
   "median_ms": 20.539719499993225
  },
  {
   "config": "scrfd__full1280",
   "recall": 0.7611086276262852,
   "median_ms": 88.36647399994035
  },
  {
   "config": "scrfd__tiled",
   "recall": 0.7612874385337506,
   "median_ms": 91.10590799991769
  },
  {
   "config": "retinaface__full640",
   "recall": 0.467411712114439,
   "median_ms": 30.86890799977482
  },
  {
   "config": "retinaface__full1280",
   "recall": 0.7916852928028609,
   "median_ms": 107.72637699983534
  },
  {
   "config": "retinaface__tiled",
   "recall": 0.7546714349575324,
   "median_ms": 124.43899150002835
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