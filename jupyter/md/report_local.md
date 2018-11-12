
# Por Métrica

## Distribuição 

|  Infrared x Redshift  |    |
|----|----|
| ANN| KNN|
|----|----|
|     ![](local/ann/redshift.png)     |     ![](local/knn/redshift.png) |
|----|----|
| RForest | LReg |
|----|----|
|         ![](local/rf/redshift.png)  |      ![](local/lr/redshift.png) |
|----|----|



## HeatMap 

|  Predito x Real  |    |
|----|----|
| ANN| KNN|
|----|----|
| ![](local/ann/hm.png) | ![](local/knn/hm.png) |
|-----------------------------|-----------------------------|
| RForest | LReg |
|----|----|
| ![](local/rf/hm.png)  | ![](local/lr/hm.png)  |
|-----------------------------|-----------------------------|

# Por Modelo

## Redes Neurais

| ![](local/ann/redshift.png) | ![](local/ann/hm.png) | ![](local/ann/mse.png) |
|----|----|----|
| ![](local/ann/curves_ex1.png) [1]| ![](local/ann/curves_ex2.png) [2] |  |
|----|----|----|
1. DESDM: 3 camadas com 15 unidades cada
1. ANNz: 2 camadas com 10 unidades cada

## Florestas Aleatóreas

| ![](local/rf/redshift.png) | ![](local/rf/hm.png) |      N/A : mean_squaded_error!       |
|----|----|----|

## K-Vizinhos mais próximos

| ![](local/knn/redshift.png) | ![](local/knn/hm.png) |      mean_squaded_error = 0.0011674860537731718       |
|----|----|----|

## Regressão Linear

| ![](local/lr/redshift.png) | ![](local/lr/hm.png) |      mean_squaded_error = 0.0024200070183046925       |
|----|----|----|
