## opencv

### 修改颜色空间

BGR RGB GRAY HSV etc

### 几何变换

1. 缩放 Scaling  

``` python
   cv.resize(img, (width, hright))
```

2. 位移 Translation  
构建位移矩阵

$$
M = \begin{bmatrix}
1  & 0 & t_x\\
0  & 1 & t_y
\end{bmatrix}
$$

``` python
cv.warpAffine(img, M, (cols, rows))
```

3. 旋转 Rotation

``` python
   M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
   dst = cv.warpAffine(img,M,(cols,rows))
```

4. 仿射 Affine

``` python
   pts1 = np.float32([[50,50],[200,50],[50,200]])
   pts2 = np.float32([[10,100],[200,50],[100,250]])
   M = cv.getAffineTransform(pts1,pts2)
   dst = cv.warpAffine(img,M,(cols,rows))
```

5. Perspective Transformation

### 阈值化 Thresholding

1. Simple Thresholding

   ``` python
      ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
   ```

   输入为灰度图，127为阈值，255为超过阈值的像素值的最大值，cv.THRESH_BINARY为阈值化方式

   ``` python
      cv.THRESH_BINARY 二值化 
         dst(x, y) = maxval if src(x, y) > thresh else 0
      cv.THRESH_BINARY_INV 反向二值化
      cv.THRESH_TRUNC 截断
         dst(x, y) = threshold if src(x, y) > thresh else src(x, y)
      cv.THRESH_TOZERO 置零
         dst(x, y) = src(x, y) if src(x, y) > thresh else 0
      cv.THRESH_TOZERO_INV 反向置零
   ```

2. Adaptive Thresholding
3. Otsu's Binarization