# State Proxy Summary

- RH time-series rows processed: `3801280`
- Mean per-UE rate proxy: `0.192116` Mbps
- Mean RLC buffer proxy: `0.040539` MB

PDCP forwarding volume is only a bitrate-window approximation; the dataset does not contain direct PDCP forwarding logs.
RLC buffer state has a stronger proxy because slice metrics expose downlink and uplink buffer fields, but it is still not a protocol-state trace.
