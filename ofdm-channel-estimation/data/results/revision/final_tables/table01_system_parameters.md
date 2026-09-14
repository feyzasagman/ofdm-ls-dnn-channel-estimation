# Table 1 — System and Simulation Parameters

| Parameter | Value |
| --- | --- |
| Subcarriers (N) | 64 |
| Cyclic prefix length | 16 |
| Modulation | QPSK |
| Pilot spacing | 4 |
| Pilot indices | [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60] |
| Number of pilots | 16 |
| Data subcarriers | 48 |
| Pilot symbol | 1+0j |
| SNR range (dB) | [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0] |
| Samples per SNR | 100 |
| Train / Val / Test per SNR | 72 / 8 / 20 |
| Random seeds | [42, 123, 999] |
| Rician K-factor | 6 dB |
| Channel model | Single-tap flat fading (AWGN / Rayleigh / Rician) |
| Split strategy | snr_stratified |
