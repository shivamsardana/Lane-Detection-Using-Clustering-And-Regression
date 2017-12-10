from __future__ import print_function
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.distance import pdist
import io, urllib
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import sys
sys.setrecursionlimit(10000)
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas
from matplotlib.pyplot import show
from decimal import Decimal

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

img=cv2.imread('um_000002.png')
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillConvexPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def select_region_of_interest(img):
    h = 20
    v1 = (0 + h, img.shape[0])
    v2 = (img.shape[1] / 3.2, img.shape[0] / 2)
    v3 = (img.shape[1] / 1.64, img.shape[0] / 2)
    v4 = (img.shape[1] /1.28, img.shape[0])

    return region_of_interest(img, np.array([[v1, v2, v3, v4]], dtype=np.int32))


imgray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgauss=cv2.GaussianBlur(imgray, (5, 5), 0)
imcanny=cv2.Canny(imgauss,90,150)
abba=select_region_of_interest(imcanny)
cv2.imshow('abba',abba)
retval, dst = cv2.threshold(abba,128, 255, cv2.THRESH_BINARY)
lines = cv2.HoughLinesP(dst,1,np.pi/180,70,100,10)
b=lines.shape[0]
print(b)
c=img.shape[0]
print(c)
m=[]
y=[]
y_cross=img.shape[0]
for i in range(0,b):
   for x1, y1, x2, y2 in lines[i]:
       c=float((y2-y1)/(x2-x1))
       if abs(c) > 0.5 and abs(c) < 2.5:
           y_cross = min(y_cross, y1)
           y_cross = min(y_cross, y2)

       m.append(c)
       y.append(b)
       b=y2-c*x2
print(y_cross)
m=np.asarray(m)
y=np.asarray(y)
a=m.shape[0]
#vc=y.shape[0]
zx=m.reshape(a,1)
#zc=y.reshape(vc,1)
distanceMatrix = pdist(zx,'euclidean')
#dend = dendrogram(linkage(distanceMatrix, method='average'),)
assignments = fcluster(linkage(distanceMatrix, method='average'),0.49,'distance')


assignments=np.array(assignments)
zxc=np.unique(assignments)
print(zxc.shape)
clu=[]
for j in range(0,lines.shape[0]):
    if assignments[j]==1:
        clu.append(j)

clu=np.asarray(clu)
print(clu.shape)
y_in=[]

for i in range(0,clu.shape[0]):
    a=clu[i]
    for x1,y1,x2,y2 in lines[a]:
        #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3, 8)
        c = float((y2 - y1) / (x2 - x1))

        b = y2 - c * x2
        y_in.append(b)
y_in=np.asarray(y_in)
y_int=y_in.reshape(y_in.shape[0],1)
print(y_int.shape)



distanceMatrix_y=pdist(y_int,'euclidean')
dend_y=dendrogram(linkage(distanceMatrix_y,method='average'),)
assignments_y = fcluster(linkage(distanceMatrix_y, method='average'),40,'distance')
assignments_y=np.array(assignments_y)
mnv=[]

zxv=np.unique(assignments_y)
print("@#@#$%^&^%$#$%^")
print(zxv.shape)



for j in range(0,clu.shape[0]):
    if assignments_y[j]==1:
        mnv.append(j)
mnv=np.asarray(mnv)
wls_x1=[]
wls_y1=[]
wls_x2=[]
wls_y2=[]
line_cl=[]
line_cl=np.asarray(line_cl)
print(assignments_y.shape)

matrix=[]
for i in range(0,mnv.shape[0]):
    a=mnv[i]
    for x1,y1,x2,y2 in lines[a]:
        wls_x1.append(x1)
        wls_y1.append(y1)
        wls_x2.append(x2)
        wls_y2.append(y2)
wls_x1=np.asarray(wls_x1)
wls_x2=np.asarray(wls_x2)
wls_y1=np.asarray(wls_y1)
wls_y2=np.asarray(wls_y2)

wls_x1=wls_x1.reshape(wls_x1.shape[0],1)
wls_y1=wls_y1.reshape(wls_y1.shape[0],1)
wls_x2=wls_x2.reshape(wls_x2.shape[0],1)
wls_y2=wls_y2.reshape(wls_y2.shape[0],1)
wls_x1=np.insert(wls_x1,0,1,axis=1)
wls_x2=np.insert(wls_x2,0,1,axis=1)
for i in range(0,mnv.shape[0]):
    a=mnv[i]
    row=[]
    for x1, y1, x2, y2 in lines[a]:
        X1=x1
        Y1=y1
        X2=x2
        Y2=y2
        '''
        if x1 >= 0 and x1 < img.shape[1] and \
                        y1 >= 0 and y1 < img.shape[0] and \
                        x2 >= 0 and x2 < img.shape[1] and \
                        y2 >= 0 and y2 < img.shape[0]:

        else:
            print('BAD LINE (%d, %d, %d, %d)' % (x1, y1, x2, y2))
        '''
    cv2.imshow('in between',img)
    for j in range(0,mnv.shape[0]):
        b=mnv[j]
        sum=0
        if(a==b):
            for k in range(0,mnv.shape[0]):
                g=mnv[k]
                for Xc1,Yc1,Xc2,Yc2 in lines[g]:
                    m=(X1-Xc1)**2
                    n=(Y1-Yc1)**2
                    o=(X2-Xc2)**2
                    p=(Y2-Yc2)**2
                    start_d=math.sqrt(m+n)
                    end_d=math.sqrt(o+p)
                    dis=(start_d+end_d)/2
                    sum=sum+dis
            row.append(sum)
        else:
            row.append(0)
    matrix.append(row)

matrix=np.asarray(matrix)
print(wls_y1.shape)
print(wls_x1.shape)
print(matrix.shape)
mod_wls = sm.OLS(wls_y1, wls_x1)
res_wls = mod_wls.fit()
print(y_cross)
print("123456765434567654567654e")
b,m=res_wls.params
bbb=float(b)
mmm=float(m)
print(bbb)
print(mmm)

YYY1 = img.shape[0] - 1
print(YYY1)
XXX1 = (YYY1 - bbb) / mmm
XXX1=int(XXX1)
print(XXX1)
x_end1 = (y_cross - bbb) / mmm
x_end1=int(x_end1)
print(x_end1)
print(y_cross)
cv2.line(img, (XXX1, YYY1), (x_end1, y_cross), [255,0,0], 2)
cv2.imshow("finally",img)
cv2.waitKey()
show()
