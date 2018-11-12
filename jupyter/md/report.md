
# Por Métrica

## Distribuição 

|  Infrared x Redshift  |    |
|----|----|
| ANN| KNN|
|----|----|
|     ![](adhafera/ann/redshift.png)     |     ![](adhafera/knn/redshift.png) |
|----|----|
| RForest | LReg |
|----|----|
|         ![](adhafera/rf/redshift.png)  |      ![](adhafera/lr/redshift.png) |
|----|----|



## HeatMap 

|  Predito x Real  |    |
|----|----|
| ANN| KNN|
|----|----|
| ![](adhafera/ann/hm.png) | ![](adhafera/knn/hm.png) |
|-----------------------------|-----------------------------|
| RForest | LReg |
|----|----|
| ![](adhafera/rf/hm.png)  | ![](adhafera/lr/hm.png)  |
|-----------------------------|-----------------------------|

# Por Modelo

## Redes Neurais

| ![](adhafera/ann/redshift.png) | ![](adhafera/ann/hm.png) | ![](adhafera/ann/mse.png) |
|----|----|----|

## Florestas Aleatóreas

| ![](adhafera/rf/redshift.png) | ![](adhafera/rf/hm.png) |      mean_squaded_error = ?       |
|----|----|----|

## K-Vizinhos mais próximos

| ![](adhafera/knn/redshift.png) | ![](adhafera/knn/hm.png) |      mean_squaded_error = 0.001171       |
|----|----|----|

## Regressão Linear

| ![](adhafera/lr/redshift.png) | ![](adhafera/lr/hm.png) |      mean_squaded_error = 0.002471       |
|----|----|----|
