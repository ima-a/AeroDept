# Independent reserved final test · completed

One frozen study, seven comparisons, 8,820 current frames per comparison. A epoch 9 remained the preselected primary model; Original epoch 4 and exploratory KD epoch 9 were fixed secondary comparisons. No training, tuning or post-test selection occurred.

## Inputs, freeze and exposure

Eight CRC-verified archives supplied 17,640 images / 8,820 RGB-range pairs. The four HDF5 streams were reused from healthy rescue storage. Archive/member hashes, sensor correspondence and underlying-path/weather-equivalent disjointness were verified before freezing. No pre-freeze scene previews or depth-distribution analysis occurred. Downloads originals remain intact; staging is external to the repository. No failed-HDD mount, repair or write was used.

| Reserved realization | Frames |
|---|---|
| Kite_training/foggy/trajectory_2007 | 2204 |
| Kite_training/sunny/trajectory_0008 | 2205 |
| PLE_training/spring/trajectory_5008 | 2206 |
| PLE_training/winter/trajectory_6009 | 2205 |

Protocol SHA-256: `6a1bb71c98ba3a5b513ff46e7babb4c33dca168886f9561072e8095cff71473f`. Manifest SHA-256: `4220cb10bd3d7671a4e08498b2f9ce34860ccdc95debce6eb93cb25cfb7da9f2`. Frozen UTC: 2026-10-30T02:49:21.424539+00:00. Exposure UTC: 2026-10-30T02:49:35.032585+00:00. Completed UTC: 2026-10-30T03:17:24.313132+00:00.

| Model | Epoch | Weights SHA-256 |
|---|---|---|
| A | 9 | 6820e7050dd5b6278ecdaf7055d9d69dbbe246a9644ebbdff77056da4278ae20 |
| Original | 4 | b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6 |
| KD | 9 | f776390515dee417c72e47e6ea41aae3710b2b9251edeaeee7a5fd7fb6fc0d45 |

The exclusive exposure ledger preceded decoding. The runner rejects existing exposure/output and has no automatic resume/retry. The original blocked proposal/specification and historical 4,436-frame test remain unchanged. The compatibility receipt permits inference from the exact rejected-teacher KD checkpoint with AST-equivalent source; it does not accept the teacher or permit training resume.

## Metrics and quality

192×320 full-frame RGB, one in-model ImageNet normalization, Euclidean range metres, FP32, valid finite GT [0.1,1000] m and near GT <10 m. No new crop, scaling or upper prediction clamp. Nonfinite predictions are fatal even outside the valid mask. AbsRel and strict ratio delta thresholds are pooled by valid pixels; RMSE comes from pooled squared error. Every current frame is scored, including reused maps. Counts agree across all seven comparisons. Classification precision/recall/F1 are not depth-regression metrics.

### pooled

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ |
|---|---|---|---|---|---|---|---|
| A_every | 0.383690 | 81.813343 | 0.406799 | 0.661703 | 0.792140 | 0.439396 | 4.144905 |
| A_fixed2 | 0.387019 | 81.975304 | 0.405759 | 0.660284 | 0.790959 | 0.442566 | 4.369061 |
| A_fixed5 | 0.412312 | 83.462253 | 0.399407 | 0.653437 | 0.785177 | 0.464057 | 7.756830 |
| A_budget_fixed | 0.386539 | 81.962581 | 0.406137 | 0.660468 | 0.791392 | 0.442450 | 4.416516 |
| A_v2 | 0.388208 | 82.298352 | 0.406031 | 0.659257 | 0.790126 | 0.439805 | 4.405438 |
| Original_every | 0.390815 | 81.798271 | 0.381188 | 0.644062 | 0.791480 | 0.487086 | 4.087260 |
| KD_every | 0.383202 | 82.173502 | 0.411403 | 0.661888 | 0.787492 | 0.436249 | 4.219446 |

### Kite_training/foggy/trajectory_2007

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ |
|---|---|---|---|---|---|---|---|
| A_every | 0.233710 | 78.364065 | 0.608260 | 0.886045 | 0.960080 | 0.182073 | 3.102119 |
| A_fixed2 | 0.235758 | 78.566267 | 0.608514 | 0.884996 | 0.958970 | 0.183720 | 3.382324 |
| A_fixed5 | 0.249707 | 80.321071 | 0.604302 | 0.880481 | 0.954849 | 0.192619 | 5.271800 |
| A_budget_fixed | 0.235239 | 78.531337 | 0.608936 | 0.885844 | 0.959255 | 0.184402 | 3.549451 |
| A_v2 | 0.241673 | 79.524608 | 0.608158 | 0.882499 | 0.957243 | 0.183913 | 4.016049 |
| Original_every | 0.282769 | 77.610310 | 0.490006 | 0.778229 | 0.904023 | 0.272358 | 2.756589 |
| KD_every | 0.230382 | 79.045396 | 0.621987 | 0.895550 | 0.961209 | 0.186724 | 3.085750 |

### Kite_training/sunny/trajectory_0008

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ |
|---|---|---|---|---|---|---|---|
| A_every | 0.544881 | 126.223733 | 0.065705 | 0.199668 | 0.400885 | 0.404106 | 4.044662 |
| A_fixed2 | 0.544626 | 126.220357 | 0.065890 | 0.200499 | 0.401088 | 0.405437 | 4.408612 |
| A_fixed5 | 0.545245 | 126.408625 | 0.065506 | 0.198532 | 0.402090 | 0.409206 | 4.813383 |
| A_budget_fixed | 0.544577 | 126.245126 | 0.065748 | 0.199383 | 0.402016 | 0.407058 | 4.346573 |
| A_v2 | 0.545369 | 126.372778 | 0.066001 | 0.198130 | 0.398508 | 0.409256 | 4.981709 |
| Original_every | 0.491769 | 126.794208 | 0.117185 | 0.289704 | 0.491811 | 0.204875 | 2.255965 |
| KD_every | 0.551861 | 126.564154 | 0.062087 | 0.190963 | 0.381596 | 0.418402 | 4.164355 |

### PLE_training/spring/trajectory_5008

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ |
|---|---|---|---|---|---|---|---|
| A_every | 0.334740 | 49.392397 | 0.493041 | 0.776523 | 0.888847 | 0.379440 | 3.365328 |
| A_fixed2 | 0.336127 | 49.549236 | 0.491859 | 0.775349 | 0.888354 | 0.379725 | 3.366251 |
| A_fixed5 | 0.350586 | 51.117740 | 0.489056 | 0.770847 | 0.884622 | 0.380790 | 3.690481 |
| A_budget_fixed | 0.335786 | 49.514489 | 0.492624 | 0.775737 | 0.888929 | 0.379598 | 3.360605 |
| A_v2 | 0.340827 | 50.083864 | 0.491641 | 0.773423 | 0.887035 | 0.378321 | 3.619280 |
| Original_every | 0.348892 | 51.077478 | 0.483851 | 0.756080 | 0.875094 | 0.443289 | 4.031056 |
| KD_every | 0.339704 | 50.012651 | 0.483985 | 0.769286 | 0.884139 | 0.396170 | 3.506637 |

### PLE_training/winter/trajectory_6009

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ |
|---|---|---|---|---|---|---|---|
| A_every | 0.440121 | 55.320550 | 0.422947 | 0.735320 | 0.877592 | 0.606827 | 5.014265 |
| A_fixed2 | 0.449838 | 55.826714 | 0.419619 | 0.731271 | 0.874442 | 0.612906 | 5.337441 |
| A_fixed5 | 0.519251 | 59.925677 | 0.402322 | 0.715413 | 0.858910 | 0.655943 | 10.533072 |
| A_budget_fixed | 0.448878 | 55.784023 | 0.420046 | 0.731742 | 0.874446 | 0.612386 | 5.376952 |
| A_v2 | 0.443212 | 55.511515 | 0.421146 | 0.733813 | 0.876556 | 0.607600 | 5.055119 |
| Original_every | 0.451437 | 53.665238 | 0.407627 | 0.716455 | 0.864294 | 0.627238 | 4.647456 |
| KD_every | 0.430639 | 55.166191 | 0.438641 | 0.741015 | 0.880190 | 0.584823 | 5.083697 |

### Pixel counts (identical across all comparisons)

| Scope | Frames | Valid pixels | Near pixels |
|---|---|---|---|
| pooled | 8820 | 377372106 | 105333081 |
| Kite_training/foggy/trajectory_2007 | 2204 | 99812690 | 22311420 |
| Kite_training/sunny/trajectory_0008 | 2205 | 87117266 | 945433 |
| PLE_training/spring/trajectory_5008 | 2206 | 93182086 | 35039499 |
| PLE_training/winter/trajectory_6009 | 2205 | 97260064 | 47036729 |

### Signed bias and floor diagnostics

Bias is P−GT in metres. Floor is P≤0.101 m on valid pixels; near-floor uses near pixels. Near ≥10 is the fraction of near-GT pixels predicted at least 10 m. Undefined denominators remain null.

**pooled**

| Model / policy | Bias m | Near bias m | Floor pixels | Floor fraction | Near floor fraction | Near predicted ≥10 fraction |
|---|---|---|---|---|---|---|
| A_every | -16.737979 | 1.845207 | 27174 | 0.000072 | 0.000032 | 0.229674 |
| A_fixed2 | -16.685233 | 1.867652 | 25039 | 0.000066 | 0.000023 | 0.229662 |
| A_fixed5 | -16.255130 | 2.012179 | 55561 | 0.000147 | 0.000086 | 0.232510 |
| A_budget_fixed | -16.697654 | 1.867820 | 25346 | 0.000067 | 0.000024 | 0.230852 |
| A_v2 | -16.612378 | 1.843760 | 25809 | 0.000068 | 0.000026 | 0.228800 |
| Original_every | -18.722901 | 2.289704 | 92462 | 0.000245 | 0.000221 | 0.281235 |
| KD_every | -16.924226 | 1.882229 | 30823 | 0.000082 | 0.000028 | 0.235455 |

**Kite_training/foggy/trajectory_2007**

| Model / policy | Bias m | Near bias m | Floor pixels | Floor fraction | Near floor fraction | Near predicted ≥10 fraction |
|---|---|---|---|---|---|---|
| A_every | -5.889047 | 0.047545 | 21475 | 0.000215 | 0.000039 | 0.100279 |
| A_fixed2 | -5.828663 | 0.061590 | 20252 | 0.000203 | 0.000026 | 0.099097 |
| A_fixed5 | -5.380968 | 0.105985 | 19695 | 0.000197 | 0.000029 | 0.098654 |
| A_budget_fixed | -5.878513 | 0.062145 | 20551 | 0.000206 | 0.000033 | 0.099810 |
| A_v2 | -5.611991 | 0.040011 | 20643 | 0.000207 | 0.000021 | 0.097969 |
| Original_every | -10.091876 | 1.196000 | 32277 | 0.000323 | 0.000107 | 0.221163 |
| KD_every | -5.905727 | 0.210666 | 25532 | 0.000256 | 0.000034 | 0.117826 |

**Kite_training/sunny/trajectory_0008**

| Model / policy | Bias m | Near bias m | Floor pixels | Floor fraction | Near floor fraction | Near predicted ≥10 fraction |
|---|---|---|---|---|---|---|
| A_every | -50.401825 | -3.685770 | 123 | 0.000001 | 0.000000 | 0.001408 |
| A_fixed2 | -50.410838 | -3.669716 | 94 | 0.000001 | 0.000000 | 0.001174 |
| A_fixed5 | -50.356889 | -3.671998 | 52 | 0.000001 | 0.000000 | 0.001325 |
| A_budget_fixed | -50.418554 | -3.696199 | 86 | 0.000001 | 0.000000 | 0.001308 |
| A_v2 | -50.395936 | -3.662462 | 78 | 0.000001 | 0.000000 | 0.001265 |
| Original_every | -52.348253 | -1.842515 | 365 | 0.000004 | 0.000002 | 0.013119 |
| KD_every | -50.795102 | -3.817250 | 130 | 0.000001 | 0.000000 | 0.000784 |

**PLE_training/spring/trajectory_5008**

| Model / policy | Bias m | Near bias m | Floor pixels | Floor fraction | Near floor fraction | Near predicted ≥10 fraction |
|---|---|---|---|---|---|---|
| A_every | -8.946472 | 1.805683 | 1848 | 0.000020 | 0.000019 | 0.219813 |
| A_fixed2 | -8.910078 | 1.802785 | 1392 | 0.000015 | 0.000014 | 0.219597 |
| A_fixed5 | -8.572572 | 1.812383 | 2995 | 0.000032 | 0.000013 | 0.220141 |
| A_budget_fixed | -8.884553 | 1.807454 | 1381 | 0.000015 | 0.000013 | 0.220834 |
| A_v2 | -8.822482 | 1.799466 | 1385 | 0.000015 | 0.000011 | 0.217761 |
| Original_every | -9.778523 | 2.328389 | 15640 | 0.000168 | 0.000072 | 0.293226 |
| KD_every | -9.353991 | 1.967258 | 1627 | 0.000017 | 0.000017 | 0.244228 |

**PLE_training/winter/trajectory_6009**

| Model / policy | Bias m | Near bias m | Floor pixels | Floor fraction | Near floor fraction | Near predicted ≥10 fraction |
|---|---|---|---|---|---|---|
| A_every | -5.183265 | 2.838527 | 3728 | 0.000038 | 0.000039 | 0.302986 |
| A_fixed2 | -5.067372 | 2.883963 | 3301 | 0.000034 | 0.000028 | 0.303685 |
| A_fixed5 | -4.229683 | 3.179450 | 32819 | 0.000337 | 0.000168 | 0.309865 |
| A_budget_fixed | -5.081951 | 2.881131 | 3328 | 0.000034 | 0.000028 | 0.305086 |
| A_v2 | -5.104324 | 2.843023 | 3703 | 0.000038 | 0.000039 | 0.303655 |
| Original_every | -6.031082 | 2.862730 | 44180 | 0.000454 | 0.000391 | 0.306186 |
| KD_every | -5.146094 | 2.726336 | 3534 | 0.000036 | 0.000033 | 0.289433 |

## Frozen contrasts and interpretation

Signed relative error change is (candidate/reference−1)×100; negative values mean lower error. These are descriptive comparisons from one run, not confidence intervals.

| Scope | Contrast | AbsRel change % | RMSE change % | Near AbsRel change % | Near RMSE change % | Call difference |
|---|---|---|---|---|---|---|
| pooled | A_v2 vs A_budget_fixed | 0.431810 | 0.409663 | -0.597948 | -0.250826 | -1194 |
| pooled | A_every vs Original_every | -1.823239 | 0.018427 | -9.790903 | 1.410371 | 0 |
| pooled | KD_every vs A_every | -0.126983 | 0.440220 | -0.716170 | 1.798364 | 0 |
| Kite_training/foggy/trajectory_2007 | A_v2 vs A_budget_fixed | 2.734691 | 1.264808 | -0.265245 | 13.145637 | -606 |
| Kite_training/foggy/trajectory_2007 | A_every vs Original_every | -17.349415 | 0.971205 | -33.149441 | 12.534700 | 0 |
| Kite_training/foggy/trajectory_2007 | KD_every vs A_every | -1.424011 | 0.869444 | 2.554550 | -0.527692 | 0 |
| Kite_training/sunny/trajectory_0008 | A_v2 vs A_budget_fixed | 0.145439 | 0.101115 | 0.540194 | 14.612356 | -533 |
| Kite_training/sunny/trajectory_0008 | A_every vs Original_every | 10.800321 | -0.449922 | 97.245514 | 79.287472 | 0 |
| Kite_training/sunny/trajectory_0008 | KD_every vs A_every | 1.280921 | 0.269696 | 3.537488 | 2.959265 | 0 |
| PLE_training/spring/trajectory_5008 | A_v2 vs A_budget_fixed | 1.501424 | 1.149916 | -0.336414 | 7.697278 | -491 |
| PLE_training/spring/trajectory_5008 | A_every vs Original_every | -4.056308 | -3.299070 | -14.403482 | -16.514986 | 0 |
| PLE_training/spring/trajectory_5008 | KD_every vs A_every | 1.482982 | 1.255768 | 4.409188 | 4.198984 | 0 |
| PLE_training/winter/trajectory_6009 | A_v2 vs A_budget_fixed | -1.262106 | -0.488505 | -0.781442 | -5.985415 | 436 |
| PLE_training/winter/trajectory_6009 | A_every vs Original_every | -2.506619 | 3.084515 | -3.254142 | 7.892666 | 0 |
| PLE_training/winter/trajectory_6009 | KD_every vs A_every | -2.154555 | -0.279026 | -3.625990 | 1.384689 | 0 |

A versus Original: pooled AbsRel is lower (-1.82%) and near RMSE is higher (+1.41%). Original remains the predeclared near-field reference; this does not guarantee it ranks first on every new path.

KD versus A: pooled AbsRel is lower (-0.13%) and near RMSE is higher (+1.80%). On validation both were worse (+1.59% and +1.42% respectively). Any improvement here is a test reversal and does not erase the negative validation result or failed teacher gate.

Inspect all four subgroup rows: pooled pixels can conceal path regressions. V2 and budget-fixed need not use equal test calls; the comparator was frozen from validation, independently of observed adaptive calls. Reuse can incidentally reduce error on correlated frames, so lower average error is not evidence of correct hazard detection. No model/policy reselection follows these outcomes.

Subgroup caution: A regresses on Kite sunny, with AbsRel +10.80% and near RMSE +79.29% versus Original. V2 near RMSE is worse than budget-fixed on Kite foggy, Kite sunny and PLE spring; its PLE winter improvement produces the small pooled near-RMSE benefit. The sunny trajectory has zero defined critical events, so event rates/delays there remain undefined rather than zero-error claims.

## Calls, freshness and events

Events are GT p10<8 m on entry or a ≥2 m drop from the previous frame, resetting per path. GT events are offline labels only. Delay is time to the next fresh update in the same path; terminal unresolved events are right-censored. No-event rates/delays stay undefined. A missed update is not a missed hazard detection.

### pooled

| Model / policy | Calls | Skip fraction | Mean age | Max age | Events | Fresh on event | Missed-update fraction | Mean delay frames | Max delay | Censored |
|---|---|---|---|---|---|---|---|---|---|---|
| A_every | 8820 | 0.000000 | 0.000000 | 0 | 20 | 20 | 0.000000 | 0.000000 | 0 | 0 |
| A_fixed2 | 4411 | 0.499887 | 0.499887 | 1 | 20 | 12 | 0.400000 | 0.400000 | 1 | 0 |
| A_fixed5 | 1765 | 0.799887 | 1.999546 | 4 | 20 | 5 | 0.750000 | 2.000000 | 4 | 0 |
| A_budget_fixed | 4912 | 0.443084 | 0.443084 | 1 | 20 | 13 | 0.350000 | 0.350000 | 1 | 0 |
| A_v2 | 3718 | 0.578458 | 1.261224 | 4 | 20 | 10 | 0.500000 | 1.250000 | 4 | 0 |
| Original_every | 8820 | 0.000000 | 0.000000 | 0 | 20 | 20 | 0.000000 | 0.000000 | 0 | 0 |
| KD_every | 8820 | 0.000000 | 0.000000 | 0 | 20 | 20 | 0.000000 | 0.000000 | 0 | 0 |

### Kite_training/foggy/trajectory_2007

| Model / policy | Calls | Skip fraction | Mean age | Max age | Events | Fresh on event | Missed-update fraction | Mean delay frames | Max delay | Censored |
|---|---|---|---|---|---|---|---|---|---|---|
| A_every | 2204 | 0.000000 | 0.000000 | 0 | 5 | 5 | 0.000000 | 0.000000 | 0 | 0 |
| A_fixed2 | 1102 | 0.500000 | 0.500000 | 1 | 5 | 4 | 0.200000 | 0.200000 | 1 | 0 |
| A_fixed5 | 441 | 0.799909 | 1.999093 | 4 | 5 | 1 | 0.800000 | 2.400000 | 4 | 0 |
| A_budget_fixed | 1227 | 0.443285 | 0.443285 | 1 | 5 | 4 | 0.200000 | 0.200000 | 1 | 0 |
| A_v2 | 621 | 0.718240 | 1.751815 | 4 | 5 | 0 | 1.000000 | 2.800000 | 4 | 0 |
| Original_every | 2204 | 0.000000 | 0.000000 | 0 | 5 | 5 | 0.000000 | 0.000000 | 0 | 0 |
| KD_every | 2204 | 0.000000 | 0.000000 | 0 | 5 | 5 | 0.000000 | 0.000000 | 0 | 0 |

### Kite_training/sunny/trajectory_0008

| Model / policy | Calls | Skip fraction | Mean age | Max age | Events | Fresh on event | Missed-update fraction | Mean delay frames | Max delay | Censored |
|---|---|---|---|---|---|---|---|---|---|---|
| A_every | 2205 | 0.000000 | 0.000000 | 0 | 0 | 0 | undefined | undefined | undefined | 0 |
| A_fixed2 | 1103 | 0.499773 | 0.499773 | 1 | 0 | 0 | undefined | undefined | undefined | 0 |
| A_fixed5 | 441 | 0.800000 | 2.000000 | 4 | 0 | 0 | undefined | undefined | undefined | 0 |
| A_budget_fixed | 1228 | 0.443084 | 0.443084 | 1 | 0 | 0 | undefined | undefined | undefined | 0 |
| A_v2 | 695 | 0.684807 | 1.434467 | 4 | 0 | 0 | undefined | undefined | undefined | 0 |
| Original_every | 2205 | 0.000000 | 0.000000 | 0 | 0 | 0 | undefined | undefined | undefined | 0 |
| KD_every | 2205 | 0.000000 | 0.000000 | 0 | 0 | 0 | undefined | undefined | undefined | 0 |

### PLE_training/spring/trajectory_5008

| Model / policy | Calls | Skip fraction | Mean age | Max age | Events | Fresh on event | Missed-update fraction | Mean delay frames | Max delay | Censored |
|---|---|---|---|---|---|---|---|---|---|---|
| A_every | 2206 | 0.000000 | 0.000000 | 0 | 9 | 9 | 0.000000 | 0.000000 | 0 | 0 |
| A_fixed2 | 1103 | 0.500000 | 0.500000 | 1 | 9 | 4 | 0.555556 | 0.555556 | 1 | 0 |
| A_fixed5 | 442 | 0.799637 | 1.999093 | 4 | 9 | 3 | 0.666667 | 1.444444 | 4 | 0 |
| A_budget_fixed | 1229 | 0.442883 | 0.442883 | 1 | 9 | 5 | 0.444444 | 0.444444 | 1 | 0 |
| A_v2 | 738 | 0.665458 | 1.452403 | 4 | 9 | 5 | 0.444444 | 0.777778 | 3 | 0 |
| Original_every | 2206 | 0.000000 | 0.000000 | 0 | 9 | 9 | 0.000000 | 0.000000 | 0 | 0 |
| KD_every | 2206 | 0.000000 | 0.000000 | 0 | 9 | 9 | 0.000000 | 0.000000 | 0 | 0 |

### PLE_training/winter/trajectory_6009

| Model / policy | Calls | Skip fraction | Mean age | Max age | Events | Fresh on event | Missed-update fraction | Mean delay frames | Max delay | Censored |
|---|---|---|---|---|---|---|---|---|---|---|
| A_every | 2205 | 0.000000 | 0.000000 | 0 | 6 | 6 | 0.000000 | 0.000000 | 0 | 0 |
| A_fixed2 | 1103 | 0.499773 | 0.499773 | 1 | 6 | 4 | 0.333333 | 0.333333 | 1 | 0 |
| A_fixed5 | 441 | 0.800000 | 2.000000 | 4 | 6 | 1 | 0.833333 | 2.500000 | 4 | 0 |
| A_budget_fixed | 1228 | 0.443084 | 0.443084 | 1 | 6 | 4 | 0.333333 | 0.333333 | 1 | 0 |
| A_v2 | 1664 | 0.245351 | 0.406349 | 4 | 6 | 5 | 0.166667 | 0.666667 | 4 | 0 |
| Original_every | 2205 | 0.000000 | 0.000000 | 0 | 6 | 6 | 0.000000 | 0.000000 | 0 | 0 |
| KD_every | 2205 | 0.000000 | 0.000000 | 0 | 6 | 6 | 0.000000 | 0.000000 | 0 | 0 |

All inference/skip reasons and individual observed/censored events remain in `results.json` and each `run/*_frames.csv`. Source indices, ages and current-frame pixel counts are retained for every comparison; the verifier checks their consistency.

## Actual measured timing

Single chronological traversal, rotated comparison order, CPU FP32 batch 1, two Torch threads. Ten synthetic warmups per model were excluded before exposure. Each comparison executed its own real fresh model calls, with no full prediction lookup cache. Inference includes transfer-in, forward and transfer-out; scheduler includes reset/decision and fresh risk-cache update. Compute is inference + scheduler. Shared I/O/decode are measured once per frame. Offline scores, events and CSV writes are excluded from per-policy compute. Whole-matrix wall includes all seven policies and cannot be reported as single-policy throughput. OS cache/load are uncontrolled; one timing pass has no repeatability interval.

**pooled**

| Model / policy | Inference s | Scheduler s | Compute s | Forward s | Transfer-in s | Transfer-out s | Median forward ms |
|---|---|---|---|---|---|---|---|
| A_every | 159.172406 | 0.022548 | 159.194954 | 158.922866 | 0.096832 | 0.152708 | 17.200667 |
| A_fixed2 | 79.616123 | 0.019938 | 79.636062 | 79.492538 | 0.048870 | 0.074715 | 17.171267 |
| A_fixed5 | 31.905507 | 0.017111 | 31.922619 | 31.855864 | 0.019660 | 0.029984 | 17.263226 |
| A_budget_fixed | 89.317520 | 0.020203 | 89.337723 | 89.179207 | 0.054711 | 0.083603 | 17.308321 |
| A_v2 | 68.299261 | 5.782189 | 74.081450 | 68.200503 | 0.035683 | 0.063075 | 17.399315 |
| Original_every | 160.139332 | 0.023054 | 160.162386 | 159.894255 | 0.096561 | 0.148516 | 17.322700 |
| KD_every | 158.855347 | 0.023094 | 158.878441 | 158.610165 | 0.096970 | 0.148213 | 17.191995 |

**Kite_training/foggy/trajectory_2007**

| Model / policy | Inference s | Scheduler s | Compute s | Forward s | Transfer-in s | Transfer-out s | Median forward ms |
|---|---|---|---|---|---|---|---|
| A_every | 38.975675 | 0.005416 | 38.981091 | 38.911875 | 0.023908 | 0.039892 | 16.672009 |
| A_fixed2 | 19.500775 | 0.004630 | 19.505405 | 19.470437 | 0.011887 | 0.018451 | 16.658092 |
| A_fixed5 | 7.868178 | 0.003945 | 7.872123 | 7.856039 | 0.004845 | 0.007295 | 16.788682 |
| A_budget_fixed | 21.859166 | 0.004794 | 21.863960 | 21.825272 | 0.013302 | 0.020592 | 16.752320 |
| A_v2 | 10.853289 | 1.150661 | 12.003950 | 10.837660 | 0.005592 | 0.010037 | 16.445444 |
| Original_every | 39.326733 | 0.005550 | 39.332283 | 39.265791 | 0.023800 | 0.037142 | 16.844051 |
| KD_every | 38.831470 | 0.005626 | 38.837096 | 38.771618 | 0.023475 | 0.036377 | 16.686308 |

**Kite_training/sunny/trajectory_0008**

| Model / policy | Inference s | Scheduler s | Compute s | Forward s | Transfer-in s | Transfer-out s | Median forward ms |
|---|---|---|---|---|---|---|---|
| A_every | 37.687099 | 0.005158 | 37.692257 | 37.627830 | 0.022385 | 0.036884 | 16.574003 |
| A_fixed2 | 18.850434 | 0.004526 | 18.854960 | 18.821242 | 0.011048 | 0.018144 | 16.632374 |
| A_fixed5 | 7.557456 | 0.003905 | 7.561360 | 7.545428 | 0.004595 | 0.007432 | 16.682015 |
| A_budget_fixed | 21.152878 | 0.004924 | 21.157802 | 21.119857 | 0.013030 | 0.019991 | 16.738367 |
| A_v2 | 11.946782 | 1.181026 | 13.127808 | 11.928780 | 0.006457 | 0.011545 | 16.677881 |
| Original_every | 37.975210 | 0.005403 | 37.980613 | 37.916981 | 0.022089 | 0.036140 | 16.774900 |
| KD_every | 37.551351 | 0.005310 | 37.556661 | 37.492554 | 0.022549 | 0.036248 | 16.541135 |

**PLE_training/spring/trajectory_5008**

| Model / policy | Inference s | Scheduler s | Compute s | Forward s | Transfer-in s | Transfer-out s | Median forward ms |
|---|---|---|---|---|---|---|---|
| A_every | 40.936956 | 0.005843 | 40.942800 | 40.874168 | 0.025137 | 0.037652 | 17.849144 |
| A_fixed2 | 20.455625 | 0.005185 | 20.460809 | 20.424185 | 0.012750 | 0.018691 | 17.800130 |
| A_fixed5 | 8.212831 | 0.004473 | 8.217303 | 8.199982 | 0.005154 | 0.007695 | 17.952339 |
| A_budget_fixed | 22.921196 | 0.005065 | 22.926261 | 22.886162 | 0.013946 | 0.021088 | 18.008481 |
| A_v2 | 13.837233 | 1.307328 | 15.144561 | 13.817501 | 0.007181 | 0.012551 | 18.057958 |
| Original_every | 41.300364 | 0.005940 | 41.306303 | 41.237435 | 0.025067 | 0.037862 | 18.026757 |
| KD_every | 40.855077 | 0.005839 | 40.860916 | 40.792344 | 0.025140 | 0.037593 | 17.859883 |

**PLE_training/winter/trajectory_6009**

| Model / policy | Inference s | Scheduler s | Compute s | Forward s | Transfer-in s | Transfer-out s | Median forward ms |
|---|---|---|---|---|---|---|---|
| A_every | 41.572676 | 0.006130 | 41.578806 | 41.508993 | 0.025403 | 0.038280 | 17.528300 |
| A_fixed2 | 20.809290 | 0.005598 | 20.814887 | 20.776674 | 0.013186 | 0.019430 | 17.560237 |
| A_fixed5 | 8.267043 | 0.004789 | 8.271832 | 8.254415 | 0.005066 | 0.007562 | 17.659176 |
| A_budget_fixed | 23.384280 | 0.005421 | 23.389701 | 23.347916 | 0.014432 | 0.021932 | 17.769138 |
| A_v2 | 31.661956 | 2.143175 | 33.805131 | 31.616562 | 0.016453 | 0.028942 | 17.708197 |
| Original_every | 41.537026 | 0.006161 | 41.543187 | 41.474048 | 0.025605 | 0.037372 | 17.647862 |
| KD_every | 41.617449 | 0.006319 | 41.623769 | 41.553649 | 0.025806 | 0.037995 | 17.554903 |

| Shared I/O s | Shared decode/resize s | Whole seven-comparison wall s |
|---|---|---|
| 33.710349 | 402.683407 | 1669.139590 |

Hardware/runtime: `{"device": "cpu", "platform": "Linux-7.2.11-arch1-1-x86_64-with-glibc2.44", "python": "3.13.12", "torch": "2.14.1+cu130", "torch_threads": 2}`. CUDA was unavailable to this runtime; historical validation CUDA timings remain separately labelled. No energy/battery savings or flight-safety claim is supported.

## Limits and evidence

Four reserved simulated paths, one seed, validation-selected checkpoints and scheduler settings, correlated neighbouring frames/weather and possible shared scenery. Pixel counts are not independent samples. The original model used different training data/budget. The teacher failed its original near-field gate; KD remains exploratory. The historical exposed test stays quarantined and separate. No training, tuning, teacher repair, KD rerun, split redraw or automatic exposed-study retry occurred.

[Ready protocol](PROTOCOL.json) · [Exposure](EXPOSURE.json) · [Completion](COMPLETION.json) · [Full comparison CSV](comparison.csv) · [Full results/reasons/events](results.json) · [Timing](timing.json) · [Inventory](INVENTORY.json) · [Preparation](prepared/PREPARED.json) · [Compatibility](COMPATIBILITY.json).

Read-only audit: `python scripts/verify_independent_test.py`. See [reproduction requirements](../../docs/independent_final_test.md), [consolidated report](../../docs/FINAL_REPORT.md), and [code guide](../../CODE_GUIDE.md).
