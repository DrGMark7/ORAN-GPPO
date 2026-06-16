# IMSI Trace Summary

- IMSI/context groups: `21183`
- Groups observed in one base station: `20763`
- Groups observed in multiple base stations: `420`
- Fraction single-BS: `0.9802`
- Groups with same-timestamp multi-BS ambiguity: `0`
- Groups with sequential non-overlapping BS intervals: `364`

## Multi-BS Examples

| dataset_group | scenario | training_id | experiment_id | imsi | base_stations | same_timestamp_multi_bs_count | sequential |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| slice_mixed | rome_slow_close | tr0 | exp1 | 1010123456026 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp1 | 1010123456027 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp2 | 1010123456009 | bs1;bs2 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp2 | 1010123456028 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp3 | 1010123456026 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp3 | 1010123456027 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp3 | 1010123456028 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp3 | 1010123456032 | bs2;bs3 | 0 | False |
| slice_mixed | rome_slow_close | tr0 | exp4 | 1010123456027 | bs1;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr0 | exp4 | 1010123456026 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr1 | exp2 | 1010123456027 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr1 | exp3 | 1010123456027 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr1 | exp4 | 1010123456026 | bs1;bs3 | 0 | False |
| slice_mixed | rome_slow_close | tr10 | exp2 | 1010123456028 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp4 | 1010123456026 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp4 | 1010123456027 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp5 | 1010123456009 | bs1;bs2 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp5 | 1010123456026 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp5 | 1010123456027 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp5 | 1010123456028 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr10 | exp5 | 1010123456032 | bs2;bs3 | 0 | True |
| slice_mixed | rome_slow_close | tr11 | exp1 | 1010123456003 | bs1;bs2 | 0 | True |
| slice_mixed | rome_slow_close | tr11 | exp1 | 1010123456008 | bs1;bs2 | 0 | True |
| slice_mixed | rome_slow_close | tr11 | exp1 | 1010123456009 | bs1;bs2 | 0 | True |
| slice_mixed | rome_slow_close | tr11 | exp1 | 1010123456011 | bs1;bs2 | 0 | True |
