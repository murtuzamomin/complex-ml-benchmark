# Benchmark Results Summary

## Performance Comparison

### Constrained Settings (64 bins, identical parameters)
| Model | R² Score | RMSE | Improvement |
|-------|----------|------|-------------|
| Our Method | 0.82 | 200.38 | Baseline |
| LightGBM | 0.44 | 370.16 | +86% |

### Against Best-Tuned LightGBM
| Model | Best R² | Best RMSE | Advantage |
|-------|----------|-----------|-----------|
| Our Method | 0.82 | 200.38 | Baseline |
| LightGBM (30 configs) | 0.65 | 235.27 | +26% |

## Reproducibility
All results reproducible using provided dataset and code.
