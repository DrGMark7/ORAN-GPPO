# Dataset Inventory

| Item | Count |
| --- | ---: |
| Dataset groups | 2 |
| Scenarios | 4 |
| Training folders | 18 |
| Experiment folders | 6 |
| Base station folders | 2144 |
| bs*.csv files | 2089 |
| ue*.csv files | 21436 |
| *_metrics.csv files | 21612 |
| Malformed folders/files | 0 |

## Dataset Groups

slice_mixed, slice_traffic

## Scenarios

rome_slow_close, rome_static_close, rome_static_far, rome_static_medium

## Observed Schemas

| Count | Example | Header |
| ---: | --- | --- |
| 21612 | `data/colosseum-oran-commag-dataset/slice_mixed/rome_static_close/tr6/exp1/bs3/slices_bs3/1010123456032_metrics.csv` | `Timestamp,num_ues,IMSI,RNTI,,slicing_enabled,slice_id,slice_prb,power_multiplier,scheduling_policy,,dl_mcs,dl_n_samples,dl_buffer [bytes],tx_brate downlink [Mbps],tx_pkts downlink,tx_errors downlink (%),dl_cqi,,ul_mcs,ul_n_samples,ul_buffer [bytes],rx_brate uplink [Mbps],rx_pkts uplink,rx_errors uplink (%),ul_rssi,ul_sinr,phr,,sum_requested_prbs,sum_granted_prbs,,dl_pmi,dl_ri,ul_n,ul_turbo_iters` |
| 21436 | `data/colosseum-oran-commag-dataset/slice_mixed/rome_static_close/tr6/exp1/bs1/ue3.csv` | `time,cc,pci,earfcn,rsrp,pl,cfo,dl_mcs,dl_snr,dl_turbo,dl_brate,dl_bler,ul_ta,ul_mcs,ul_buff,ul_brate,ul_bler,rf_o,rf_u,rf_l,is_attached` |
| 2089 | `data/colosseum-oran-commag-dataset/slice_mixed/rome_static_close/tr6/exp1/bs1/bs1.csv` | `time,nof_ue,dl_brate,ul_brate` |

## Malformed CSV Files

None found.
