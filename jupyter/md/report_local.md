
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

## Florestas Aleatóreas

| ![](local/rf/redshift.png) | ![](local/rf/hm.png) |      N/A : mean_squaded_error!       |
|----|----|----|

## K-Vizinhos mais pŕximos

| ![](local/knn/redshift.png) | ![](local/knn/hm.png) |      mean_squaded_error = 0.00119656851409       |
|----|----|----|

## Regressão Linear

| ![](local/lr/redshift.png) | ![](local/lr/hm.png) |      mean_squaded_error = 0.002415116132477088       |
|----|----|----|
