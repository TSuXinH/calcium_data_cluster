<center>
    <h3>
        Essay in neural cluster
    </h3>
</center>

##### Silhouette score:

$a(i)=\frac{1}{|C_I|-1}\sum_{j\in C_I,j\neq i}d(i,j)$

$b(i)=\min_{J\neq I}\frac{1}{C_J}\sum_{j\in C_J}d(i,j)$

$s(i)=\frac{b(i)-a(i)}{\max{[a(i)-b(i)]}}$



##### Calinski-Harabasz score

![image-20221019212609498](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221019212609498.png)



##### Dynamic time warping is obviously not suitable for neural response data !

