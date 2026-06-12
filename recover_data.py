import re, csv, os
from pathlib import Path

# 이전에 실행되었던 9시간 분량의 터미널 출력을 텍스트로 바로 저장
RAW_LOG = """
SpMV — laplacian_2d  N=99,856  nnz=498,016  nnz/row=5.0
  scipy_cpu                           384.5      19.70     2.59
  jax_1gpu                            101.4      74.71     9.83
  kk_1gpu                             983.6       7.70     1.01
  custom_1gpu                          53.2     142.24    18.71
  --- Batch SpMV (k=8) ---
  jax_vmap                             85.0     154.91    93.75
  kokkos_batch                        299.5      43.96    26.60
  custom_batch                         96.3     136.78    82.78
  --- Batch SpMV (k=32) ---
  jax_vmap                             85.0     380.48   375.01
  kokkos_batch                        308.7     104.74   103.24
  custom_batch                        236.5     136.71   134.74
  --- Batch SpMV (k=128) ---
  jax_vmap                            938.0     116.24   135.92
  kokkos_batch                       1208.3      90.23   105.51
  custom_batch                       1200.6      90.81   106.19
  --- Batch SpMV (k=256) ---
  jax_vmap                           1757.2     120.24   145.11
  kokkos_batch                       2374.7      88.97   107.38
  custom_batch                       2375.2      88.95   107.35
  --- Batch SpMV (k=512) ---
  jax_vmap                           2924.5     142.17   174.38
  kokkos_batch                       4758.0      87.39   107.18
  custom_batch                       4754.9      87.44   107.25

SpMV — laplacian_2d  N=299,209  nnz=1,493,857  nnz/row=5.0
  scipy_cpu                          1247.2      18.21     2.40
  jax_1gpu                            145.4     156.21    20.55
  kk_1gpu                             995.3      22.82     3.00
  custom_1gpu                         123.9     183.32    24.11
  --- Batch SpMV (k=8) ---
  jax_vmap                             86.0     458.86   277.88
  kokkos_batch                        955.4      41.31    25.02
  custom_batch                        266.2     148.25    89.78
  --- Batch SpMV (k=32) ---
  jax_vmap                            818.2     118.46   116.85
  kokkos_batch                        839.7     115.42   113.86
  custom_batch                        653.3     148.35   146.34
  --- Batch SpMV (k=128) ---
  jax_vmap                           2349.1     139.08   162.80
  kokkos_batch                       3468.3      94.20   110.26
  custom_batch                       3470.3      94.14   110.20
  --- Batch SpMV (k=256) ---
  jax_vmap                           4028.4     157.16   189.86
  kokkos_batch                       6965.8      90.89   109.80
  custom_batch                       6960.6      90.95   109.88
  --- Batch SpMV (k=512) ---
  jax_vmap                           7526.4     165.53   203.25
  kokkos_batch                      14081.5      88.48   108.63
  custom_batch                      14096.9      88.38   108.51

SpMV — laplacian_2d  N=1,000,000  nnz=4,996,000  nnz/row=5.0
  scipy_cpu                          6385.2      11.90     1.56
  jax_1gpu                            198.7     382.33    50.30
  kk_1gpu                            1047.6      72.50     9.54
  custom_1gpu                         412.7     184.05    24.21
  --- Batch SpMV (k=8) ---
  jax_vmap                            785.4     168.00   101.78
  kokkos_batch                       3127.8      42.19    25.56
  custom_batch                        826.4     159.68    96.73
  --- Batch SpMV (k=32) ---
  jax_vmap                           2129.9     152.10   150.12
  kokkos_batch                       2468.4     131.24   129.54
  custom_batch                       2127.9     152.24   150.26
  --- Batch SpMV (k=128) ---
  jax_vmap                           6668.3     163.75   191.80
  kokkos_batch                      10984.4      99.41   116.44
  custom_batch                      10982.9      99.42   116.45
  --- Batch SpMV (k=256) ---
  jax_vmap                          12728.3     166.24   200.97
  kokkos_batch                      22150.1      95.53   115.48
  custom_batch                      22145.5      95.55   115.51
  --- Batch SpMV (k=512) ---
  jax_vmap                          23949.3     173.87   213.61
  kokkos_batch                      45204.0      92.11   113.17
  custom_batch                      45230.6      92.06   113.11

SpMV — laplacian_2d  N=2,999,824  nnz=14,992,192  nnz/row=5.0
  scipy_cpu                         20701.2      11.01     1.45
  jax_1gpu                            610.3     373.43    49.13
  kk_1gpu                            1187.3     191.95    25.25
  custom_1gpu                        1188.9     191.70    25.22
  --- Batch SpMV (k=8) ---
  jax_vmap                           1997.8     198.16   120.07
  kokkos_batch                       9325.6      42.45    25.72
  custom_batch                       2422.8     163.40    99.01
  --- Batch SpMV (k=32) ---
  jax_vmap                           5435.4     178.80   176.53
  kokkos_batch                       7175.7     135.44   133.72
  custom_batch                       6318.1     153.82   151.87
  --- Batch SpMV (k=128) ---
  jax_vmap                          18808.8     174.16   204.05
  kokkos_batch                      32601.1     100.48   117.73
  custom_batch                      32560.1     100.61   117.87
  --- Batch SpMV (k=256) ---
  jax_vmap                          37140.5     170.91   206.67
  kokkos_batch                      66668.0      95.21   115.14
  custom_batch                      66680.3      95.19   115.12
  --- Batch SpMV (k=512) ---
  jax_vmap                          71101.4     175.68   215.92
  kokkos_batch                     137948.2      90.55   111.29
  custom_batch                     137930.7      90.56   111.30

SpMV — laplacian_3d  N=97,336  nnz=668,656  nnz/row=6.9
  scipy_cpu                           512.5      18.69     2.61
  jax_1gpu                            125.4      76.38    10.66
  kk_1gpu                             983.6       9.74     1.36
  custom_1gpu                          52.2     183.46    25.61
  --- Batch SpMV (k=8) ---
  jax_vmap                             86.0     174.76   124.38
  kokkos_batch                        292.9      51.33    36.53
  custom_batch                         94.2     159.56   113.56
  --- Batch SpMV (k=32) ---
  jax_vmap                             85.0     396.75   503.51
  kokkos_batch                        275.5     122.42   155.36
  custom_batch                        246.8     136.64   173.41
  --- Batch SpMV (k=128) ---
  jax_vmap                           1158.1      93.66   147.80
  kokkos_batch                       1040.4     104.26   164.53
  custom_batch                       1040.4     104.26   164.53
  --- Batch SpMV (k=256) ---
  jax_vmap                           2156.5      96.52   158.75
  kokkos_batch                       2079.7     100.08   164.61
  custom_batch                       2081.8      99.98   164.45
  --- Batch SpMV (k=512) ---
  jax_vmap                           3766.3     108.19   181.80
  kokkos_batch                       4171.3      97.69   164.15
  custom_batch                       4167.2      97.79   164.31

SpMV — laplacian_3d  N=300,763  nnz=2,078,407  nnz/row=6.9
  scipy_cpu                          1841.7      16.16     2.26
  jax_1gpu                            183.8     161.87    22.61
  kk_1gpu                             998.9      29.79     4.16
  custom_1gpu                         141.3     210.55    29.42
  --- Batch SpMV (k=8) ---
  jax_vmap                             84.0     554.92   396.04
  kokkos_batch                        967.7      48.15    34.37
  custom_batch                        275.5     169.16   120.73
  --- Batch SpMV (k=32) ---
  jax_vmap                            971.8     107.37   136.88
  kokkos_batch                        743.4     140.35   178.93
  custom_batch                        701.4     148.75   189.64
  --- Batch SpMV (k=128) ---
  jax_vmap                           3105.8     107.97   171.32
  kokkos_batch                       2980.4     112.51   178.53
  custom_batch                       2986.5     112.28   178.16
  --- Batch SpMV (k=256) ---
  jax_vmap                           5662.7     113.60   187.92
  kokkos_batch                       6053.4     106.27   175.79
  custom_batch                       6048.8     106.35   175.93
  --- Batch SpMV (k=512) ---
  jax_vmap                          10390.5     121.19   204.83
  kokkos_batch                      12435.5     101.26   171.15
  custom_batch                      12427.3     101.33   171.26

SpMV — laplacian_3d  N=1,000,000  nnz=6,940,000  nnz/row=6.9
  scipy_cpu                          8423.9      11.79     1.65
  jax_1gpu                            349.7     283.90    39.69
  kk_1gpu                            1059.8      93.67    13.10
  custom_1gpu                         416.8     238.21    33.30
  --- Batch SpMV (k=8) ---
  jax_vmap                           1028.1     151.04   108.01
  kokkos_batch                       3149.8      49.30    35.25
  custom_batch                        846.8     183.36   131.12
  --- Batch SpMV (k=32) ---
  jax_vmap                           2808.8     123.64   158.13
  kokkos_batch                       2312.2     150.20   192.09
  custom_batch                       2276.4     152.56   195.12
  --- Batch SpMV (k=128) ---
  jax_vmap                           9024.5     123.58   196.87
  kokkos_batch                       9814.0     113.64   181.03
  custom_batch                       9821.7     113.55   180.89
  --- Batch SpMV (k=256) ---
  jax_vmap                          17511.4     122.16   202.91
  kokkos_batch                      20095.5     106.46   176.82
  custom_batch                      20098.0     106.44   176.80
  --- Batch SpMV (k=512) ---
  jax_vmap                          33547.3     124.82   211.84
  kokkos_batch                      40986.6     102.16   173.39
  custom_batch                      40999.4     102.13   173.33

SpMV — laplacian_3d  N=2,985,984  nnz=20,777,472  nnz/row=7.0
  scipy_cpu                         32192.5       9.23     1.29
  jax_1gpu                            814.1     364.96    51.05
  kk_1gpu                            1218.6     243.82    34.10
  custom_1gpu                        1194.0     248.84    34.80
  --- Batch SpMV (k=8) ---
  jax_vmap                           2654.2     174.94   125.25
  kokkos_batch                       9360.4      49.60    35.52
  custom_batch                       2478.1     187.37   134.15
  --- Batch SpMV (k=32) ---
  jax_vmap                           7353.3     141.11   180.84
  kokkos_batch                       6890.0     150.60   193.00
  custom_batch                       6775.3     153.15   196.27
  --- Batch SpMV (k=128) ---
  jax_vmap                          25885.7     128.68   205.48
  kokkos_batch                      29408.3     113.26   180.87
  custom_batch                      29401.6     113.29   180.91
  --- Batch SpMV (k=256) ---
  jax_vmap                          51207.2     124.76   207.75
  kokkos_batch                      60059.1     106.37   177.13
  custom_batch                      60052.0     106.38   177.15
  --- Batch SpMV (k=512) ---
  jax_vmap                          99263.5     125.97   214.34
  kokkos_batch                     122466.3     102.10   173.73
  custom_batch                     122500.1     102.07   173.68

SpMV — random_sparse  N=100,000  nnz=1,899,910  nnz/row=19.0
  scipy_cpu                          2860.0       8.53     1.33
  jax_1gpu                            228.4     106.85    16.64
  kk_1gpu                            1002.5      24.34     3.79
  custom_1gpu                          62.5     390.61    60.83
  --- Batch SpMV (k=8) ---
  jax_vmap                             84.0     357.27   362.03
  kokkos_batch                        346.1      86.67    87.83
  custom_batch                        130.0     230.68   233.75
  --- Batch SpMV (k=32) ---
  jax_vmap                           1111.0      44.28   109.44
  kokkos_batch                        718.8      68.44   169.15
  custom_batch                        477.2     103.10   254.82
  --- Batch SpMV (k=128) ---
  jax_vmap                           3563.5      35.36   136.49
  kokkos_batch                       2926.6      43.05   166.19
  custom_batch                       2929.2      43.02   166.05
  --- Batch SpMV (k=256) ---
  jax_vmap                           6603.8      34.59   147.30
  kokkos_batch                       6190.6      36.89   157.13
  custom_batch                       6195.2      36.87   157.02
  --- Batch SpMV (k=512) ---
  jax_vmap                          13009.9      33.30   149.54
  kokkos_batch                      13083.6      33.11   148.70
  custom_batch                      13085.7      33.10   148.67

SpMV — random_sparse  N=300,000  nnz=5,699,910  nnz/row=19.0
  scipy_cpu                         20278.8       3.61     0.56
  jax_1gpu                            611.3     119.74    18.65
  kk_1gpu                            1056.8      69.27    10.79
  custom_1gpu                         155.6     470.29    73.24
  --- Batch SpMV (k=8) ---
  jax_vmap                           1087.5      82.76    83.86
  kokkos_batch                       1047.0      85.96    87.10
  custom_batch                        358.9     250.75   254.10
  --- Batch SpMV (k=32) ---
  jax_vmap                           2816.0      52.41   129.54
  kokkos_batch                       2184.2      67.58   167.02
  custom_batch                       1644.5      89.75   221.82
  --- Batch SpMV (k=128) ---
  jax_vmap                           9471.0      39.91   154.07
  kokkos_batch                       9557.5      39.55   152.67
  custom_batch                       9558.0      39.55   152.67
  --- Batch SpMV (k=256) ---
  jax_vmap                          18501.6      37.03   157.73
  kokkos_batch                      19956.7      34.33   146.23
  custom_batch                      19951.1      34.34   146.28
  --- Batch SpMV (k=512) ---
  jax_vmap                          37802.0      34.38   154.40
  kokkos_batch                      41116.2      31.61   141.96
  custom_batch                      41110.0      31.61   141.98

SpMV — random_sparse  N=1,000,000  nnz=18,999,910  nnz/row=19.0
  scipy_cpu                         80117.8       3.05     0.47
  jax_1gpu                           1641.0     148.69    23.16
  kk_1gpu                            1228.8     198.57    30.92
  custom_1gpu                         455.7     535.46    83.39
  --- Batch SpMV (k=8) ---
  jax_vmap                           3479.6      86.22    87.37
  kokkos_batch                       3711.0      80.84    81.92
  custom_batch                       1548.3     193.76   196.34
  --- Batch SpMV (k=32) ---
  jax_vmap                           8693.8      56.59   139.87
  kokkos_batch                       7970.8      61.73   152.56
  custom_batch                       6599.7      74.55   184.25
  --- Batch SpMV (k=128) ---
  jax_vmap                          30503.9      41.31   159.45
  kokkos_batch                      33991.2      37.07   143.10
  custom_batch                      33982.0      37.08   143.13
  --- Batch SpMV (k=256) ---
  jax_vmap                          60282.9      37.89   161.37
  kokkos_batch                      69276.7      32.97   140.42
  custom_batch                      69279.2      32.97   140.42
  --- Batch SpMV (k=512) ---
  jax_vmap                         124715.0      34.74   156.00
  kokkos_batch                     140249.6      30.89   138.72
  custom_batch                     140241.9      30.89   138.73

SpMV — random_sparse  N=3,000,000  nnz=56,999,910  nnz/row=19.0
  scipy_cpu                        278460.4       2.63     0.41
  jax_1gpu                           3881.0     188.61    29.37
  kk_1gpu                            1952.8     374.85    58.38
  custom_1gpu                        1616.9     452.72    70.51
  --- Batch SpMV (k=8) ---
  jax_vmap                          10551.3      85.30    86.43
  kokkos_batch                      11329.5      79.44    80.50
  custom_batch                       5508.6     163.38   165.56
  --- Batch SpMV (k=32) ---
  jax_vmap                          26775.6      55.12   136.24
  kokkos_batch                      25272.3      58.40   144.35
  custom_batch                      21021.7      70.21   173.53
  --- Batch SpMV (k=128) ---
  jax_vmap                          94834.7      39.86   153.87
  kokkos_batch                     104848.9      36.05   139.17
  custom_batch                     104846.3      36.05   139.17
  --- Batch SpMV (k=256) ---
  jax_vmap                         187981.8      36.45   155.25
  kokkos_batch                     211185.7      32.45   138.19
  custom_batch                     211194.9      32.44   138.18
  --- Batch SpMV (k=512) ---
  jax_vmap                         377061.4      34.47   154.80
  kokkos_batch                     424333.8      30.63   137.55
  custom_batch                     424240.6      30.63   137.58

SpMV — lattice_gauge  N=99,856  nnz=499,280  nnz/row=5.0
  scipy_cpu                           401.4      18.91     2.49
  jax_1gpu                            102.4      74.11     9.75
  kk_1gpu                             981.5       7.73     1.02
  custom_1gpu                          53.2     142.52    18.75
  --- Batch SpMV (k=8) ---
  jax_vmap                             86.0     153.24    92.87
  kokkos_batch                        300.0      43.93    26.63
  custom_batch                         96.3     136.94    82.99
  --- Batch SpMV (k=32) ---
  jax_vmap                             86.0     376.13   371.49
  kokkos_batch                        307.2     105.32   104.02
  custom_batch                        237.6     136.19   134.50
  --- Batch SpMV (k=128) ---
  jax_vmap                            915.5     119.11   139.62
  kokkos_batch                       1196.5      91.13   106.82
  custom_batch                       1193.0      91.41   107.14
  --- Batch SpMV (k=256) ---
  jax_vmap                           1740.8     121.38   146.85
  kokkos_batch                       2353.7      89.77   108.61
  custom_batch                       2347.0      90.03   108.92
  --- Batch SpMV (k=512) ---
  jax_vmap                           2846.7     146.06   179.60
  kokkos_batch                       4715.5      88.18   108.42
  custom_batch                       4715.0      88.19   108.43

SpMV — lattice_gauge  N=299,209  nnz=1,496,045  nnz/row=5.0
  scipy_cpu                          1206.3      18.85     2.48
  jax_1gpu                            143.4     158.62    20.87
  kk_1gpu                             993.3      22.89     3.01
  custom_1gpu                         124.9     182.02    23.95
  --- Batch SpMV (k=8) ---
  jax_vmap                            100.4     393.57   238.53
  kokkos_batch                        955.4      41.34    25.05
  custom_batch                        266.2     148.35    89.91
  --- Batch SpMV (k=32) ---
  jax_vmap                            799.7     121.22   119.72
  kokkos_batch                        838.7     115.59   114.17
  custom_batch                        654.3     148.16   146.33
  --- Batch SpMV (k=128) ---
  jax_vmap                           2414.6     135.32   158.61
  kokkos_batch                       3452.4      94.64   110.93
  custom_batch                       3454.0      94.60   110.88
  --- Batch SpMV (k=256) ---
  jax_vmap                           4122.6     153.57   185.80
  kokkos_batch                       6928.4      91.38   110.56
  custom_batch                       6919.7      91.50   110.70
  --- Batch SpMV (k=512) ---
  jax_vmap                           7590.9     164.13   201.81
  kokkos_batch                      13998.1      89.01   109.44
  custom_batch                      14005.8      88.96   109.38

SpMV — lattice_gauge  N=1,000,000  nnz=5,000,000  nnz/row=5.0
  scipy_cpu                          6164.0      12.33     1.62
  jax_1gpu                            197.1     385.55    50.73
  kk_1gpu                            1048.6      72.48     9.54
  custom_1gpu                         413.7     183.71    24.17
  --- Batch SpMV (k=8) ---
  jax_vmap                            803.8     164.21    99.52
  kokkos_batch                       3132.4      42.14    25.54
  custom_batch                        826.4     159.74    96.81
  --- Batch SpMV (k=32) ---
  jax_vmap                           2150.4     150.67   148.81
  kokkos_batch                       2458.1     131.81   130.18
  custom_batch                       2127.9     152.26   150.38
  --- Batch SpMV (k=128) ---
  jax_vmap                           6668.3     163.76   191.95
  kokkos_batch                      10934.3      99.87   117.06
  custom_batch                      10941.4      99.80   116.99
  --- Batch SpMV (k=256) ---
  jax_vmap                          12774.4     165.64   200.40
  kokkos_batch                      22024.7      96.07   116.23
  custom_batch                      22022.1      96.09   116.25
  --- Batch SpMV (k=512) ---
  jax_vmap                          23983.1     173.62   213.48
  kokkos_batch                      44939.8      92.66   113.93
  custom_batch                      44949.5      92.64   113.91

SpMV — lattice_gauge  N=2,999,824  nnz=14,999,120  nnz/row=5.0
  scipy_cpu                         20156.9      11.31     1.49
  jax_1gpu                            538.6     423.28    55.69
  kk_1gpu                            1184.8     192.43    25.32
  custom_1gpu                        1189.9     191.60    25.21
  --- Batch SpMV (k=8) ---
  jax_vmap                           1905.7     207.79   125.93
  kokkos_batch                       9333.8      42.42    25.71
  custom_batch                       2425.9     163.23    98.93
  --- Batch SpMV (k=32) ---
  jax_vmap                           5446.7     178.45   176.24
  kokkos_batch                       7132.7     136.27   134.58
  custom_batch                       6320.1     153.79   151.89
  --- Batch SpMV (k=128) ---
  jax_vmap                          18905.1     173.28   203.11
  kokkos_batch                      32409.6     101.08   118.48
  custom_batch                      32401.4     101.10   118.51
  --- Batch SpMV (k=256) ---
  jax_vmap                          37150.7     170.86   206.71
  kokkos_batch                      66298.4      95.74   115.83
  custom_batch                      66289.7      95.76   115.85
  --- Batch SpMV (k=512) ---
  jax_vmap                          71149.6     175.56   215.87
  kokkos_batch                     137155.6      91.07   111.98
  custom_batch                     137160.2      91.07   111.98

SpMV — power_law  N=100,000  nnz=699,988  nnz/row=7.0
  scipy_cpu                          1165.3       8.58     1.20
  jax_1gpu                            167.9      59.55     8.34
  kk_1gpu                             990.2      10.10     1.41
  custom_1gpu                          54.3     184.25    25.80
  --- Batch SpMV (k=8) ---
  jax_vmap                             85.0     183.54   131.77
  kokkos_batch                        303.1      51.47    36.95
  custom_batch                        101.4     153.88   110.48
  --- Batch SpMV (k=32) ---
  jax_vmap                             85.0     409.45   527.10
  kokkos_batch                       2485.2      14.00    18.03
  custom_batch                        273.4     127.28   163.85
  --- Batch SpMV (k=128) ---
  jax_vmap                           1487.9      75.01   120.44
  kokkos_batch                      10496.0      10.63    17.07
  custom_batch                      10500.6      10.63    17.07
  --- Batch SpMV (k=256) ---
  jax_vmap                           2607.1      82.08   137.47
  kokkos_batch                      21159.4      10.11    16.94
  custom_batch                      21170.7      10.11    16.93
  --- Batch SpMV (k=512) ---
  jax_vmap                           4606.0      90.93   155.62
  kokkos_batch                      42893.3       9.76    16.71
  custom_batch                      42861.6       9.77    16.72

SpMV — power_law  N=300,000  nnz=2,099,988  nnz/row=7.0
  scipy_cpu                          5273.1       5.69     0.80
  jax_1gpu                            219.6     136.58    19.12
  kk_1gpu                            1007.6      29.77     4.17
  custom_1gpu                         141.3     212.30    29.72
  --- Batch SpMV (k=8) ---
  jax_vmap                             85.0     550.64   395.33
  kokkos_batch                        968.7      48.31    34.69
  custom_batch                        280.6     166.80   119.75
  --- Batch SpMV (k=32) ---
  jax_vmap                           1211.4      86.18   110.95
  kokkos_batch                       4859.9      21.48    27.65
  custom_batch                        836.6     124.79   160.65
  --- Batch SpMV (k=128) ---
  jax_vmap                           3600.4      92.99   149.32
  kokkos_batch                      19965.4      16.77    26.93
  custom_batch                      19962.4      16.77    26.93
  --- Batch SpMV (k=256) ---
  jax_vmap                           6651.9      96.51   161.64
  kokkos_batch                      40319.0      15.92    26.67
  custom_batch                      40314.9      15.92    26.67
  --- Batch SpMV (k=512) ---
  jax_vmap                          12726.3      98.72   168.97
  kokkos_batch                      81139.7      15.48    26.50
  custom_batch                      81177.6      15.48    26.49

SpMV — power_law  N=1,000,000  nnz=6,999,988  nnz/row=7.0
  scipy_cpu                         30319.1       3.30     0.46
  jax_1gpu                            589.3     169.69    23.76
  kk_1gpu                            1083.4      92.30    12.92
  custom_1gpu                         420.9     237.61    33.26
  --- Batch SpMV (k=8) ---
  jax_vmap                           1249.3     124.87    89.65
  kokkos_batch                       3288.1      47.44    34.06
  custom_batch                        967.7     161.21   115.74
  --- Batch SpMV (k=32) ---
  jax_vmap                           3063.8     113.58   146.22
  kokkos_batch                      10251.3      33.95    43.70
  custom_batch                       2949.1     118.00   151.91
  --- Batch SpMV (k=128) ---
  jax_vmap                          10657.8     104.71   168.14
  kokkos_batch                      41596.9      26.83    43.08
  custom_batch                      41593.3      26.83    43.08
  --- Batch SpMV (k=256) ---
  jax_vmap                          20893.7     102.42   171.53
  kokkos_batch                      83842.6      25.52    42.75
  custom_batch                      83758.6      25.55    42.79
  --- Batch SpMV (k=512) ---
  jax_vmap                          41098.2     101.90   174.41
  kokkos_batch                     168435.7      24.86    42.56
  custom_batch                     168355.8      24.88    42.58
"""

def main():
    rows = []
    current_matrix = ""
    current_N = 0
    current_nnz = 0
    current_k = ""

    for line in RAW_LOG.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("SpMV —"):
            parts = line.split()
            current_matrix = parts[2]
            current_N = int(parts[3].split('=')[1].replace(',', ''))
            current_nnz = int(parts[4].split('=')[1].replace(',', ''))
            current_k = ""
        elif line.startswith("--- Batch SpMV (k="):
            m = re.search(r'k=(\d+)', line)
            if m:
                current_k = int(m.group(1))
        elif re.match(r'^(scipy_cpu|jax_1gpu|kk_1gpu|custom_1gpu|jax_vmap|kokkos_batch|custom_batch)\s+', line):
            parts = line.split()
            if len(parts) >= 4:
                rows.append({
                    'backend': parts[0], 'matrix': current_matrix,
                    'N': current_N, 'nnz': current_nnz, 'cuda_aware': 1,
                    'elapsed_us': float(parts[1]),
                    'elapsed_s': float(parts[1]) / 1e6,
                    'bw_GB_s': float(parts[2]),
                    'gflops': float(parts[3]),
                    'k': current_k, 'n_ranks': 1, 'overlap': 1
                })

    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spmv_recovered.csv"
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['backend','matrix','N','nnz','cuda_aware','elapsed_us','elapsed_s','bw_GB_s','gflops','k','n_ranks','overlap'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ {len(rows)}개의 측정 결과 CSV 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
