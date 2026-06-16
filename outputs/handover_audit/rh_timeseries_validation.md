# RH Time-Series Validation

The validation joins reconstructed RH time series with `bs*.csv` on exact timestamps within the same base-station folder.

| Group | N corr | N MAE | Lambda corr | Lambda MAE |
| --- | ---: | ---: | ---: | ---: |
| ALL | 0.9749 | 0.1794 | 0.9858 | 1.46e+06 |
| rome_slow_close | 0.9608 | 0.3798 | 0.973 | 9.671e+05 |
| rome_static_close | 0.9843 | 0.03807 | 0.9876 | 1.839e+06 |
| rome_static_far | 0.9678 | 0.2853 | 0.9847 | 1.33e+06 |
| rome_static_medium | 0.9874 | 0.05503 | 0.9875 | 1.563e+06 |
