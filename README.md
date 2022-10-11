﻿# Гибридные суперкомпьютерные вычисления

## Описание алгоритма выполнения бизнес-логики 

### Генерация данных 

Программа генерации файла с исходными матрицами А и B GenerateFile.cpp. Программа запрашивает название для генерируемого файла, его размер в мегабайтах, а также коэффициент свертки, определяющий размер ядра свертки (матрицы B) относительно размера матрицы А. 

Введенный размер файла в мегабайтах определяет размерность матрицы A, размерность матрицы B зависит от размерности A по следующему соотношению: 

M = cf * N, где cf - коэффициент, принимающий значения от 0 до 1 

После вычисления размеров матрицы, выполняется запись в файл элементов для двух матриц с указанием размерностей перед началом каждой из матриц. 

### Основная программа 

Код для реализации бизнес-логики в MatrixConvolution.cpp. Программа запрашивает название файла с исходными данными. Далее происходит считывание из файла матрицы А (NxN), ее создание и заполнение, затем, аналогично, заполнение матрицы B(MxM). 

Вычисляется размерность матрицы свертки C (K x K): 
K =N−M+1 
K =N−M+1 

Вычисление результата свертки  происходит согласно формуле 
Ci,j=∑M−1k=0∑M−1l=0Ai+k,j+l⋅Bk,l
Ci,j=∑k=0M−1∑l=0M−1Ai+k,j+l⋅Bk,l
 
Полученная матрица С записывается в файл. В конец файла также записывается время выполнения программы и размер исходных данных, на основе которых был построен график. 

## График зависимости времени выполнения от размера данных (5 точек)

Для всех исходных данных был задан коэффициент cf = 0.5, поэтому матрица B в два раза меньше по размерности матрицы А.
То есть ядро свертки значительно увеличивается с увеличением объема данных, что, исходя из алгоритма работы, дает зависимость близкую к x^4. 
![](./img/sequential_chart.png)