
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

| ![](adhafera/ann/redshift.png) | ![](adhafera/ann/hm.png) | ![](adhafera/ann/mse.png) mean_squaded_error = 0.001099|
|----|----|----|
| ![](adhafera/ann/curves.png) [1]|  | |
|----|----|----|
1. ANN: 2 camadas, com 175 unidades na camada 1 e 151 na camada 2 [dropout=0.1].


## Florestas Aleatóreas

| ![](adhafera/rf/redshift.png) | ![](adhafera/rf/hm.png) |      mean_squaded_error = 0.001070       |
|----|----|----|

## K-Vizinhos mais próximos

| ![](adhafera/knn/redshift.png) | ![](adhafera/knn/hm.png) |      mean_squaded_error = 0.001171       |
|----|----|----|

## Regressão Linear

| ![](adhafera/lr/redshift.png) | ![](adhafera/lr/hm.png) |      mean_squaded_error = 0.002471       |
|----|----|----|
