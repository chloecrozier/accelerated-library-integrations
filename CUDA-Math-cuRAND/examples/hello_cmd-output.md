// cuRAND accepts seeds from 0 to 2^192 to produce outputs from 0 to 1 that are statistically random. The seed itself is the random factor. cuRAND itself is deterministic. Here are the outputs of compiled code with cuRAND where you can pass a seed value at CLI.

$ ./curand_cmd 11
cuRAND hello world (seed=11):
0.489410
0.314093
0.913334
0.354308
0.972853
0.473386
0.743827
0.502127
0.443645
0.557737

$ ./curand_cmd 12
cuRAND hello world (seed=12):
0.336982
0.951115
0.474680
0.914804
0.922780
0.439865
0.782349
0.541669
0.586030
0.474092

$ ./curand_cmd 13
cuRAND hello world (seed=13):
0.561971
0.702638
0.604982
0.308525
0.161740
0.503189
0.965314
0.669302
0.011212
0.491659
