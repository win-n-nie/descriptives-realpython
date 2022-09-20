from cmath import nan
import math
from multiprocessing.spawn import get_command_line
import statistics
from subprocess import STD_ERROR_HANDLE
import numpy as np
import scipy.stats
import pandas as pd

x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
x_with_nan

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y

y_with_nan
z
z_with_nan

mean_ = sum(x) / len(x)
mean_

mean_ = statistics.mean(x
mean_

mean_ = statistics.mean(x_with_nan)
mean_

mean_ = statistics.fmean(x_with_nan)
mean_

mean_ = np.mean(y)
mean_

mean_ = y.mean()
mean_

np.mean(y_with_nan)
nan

y_with_nan.mean()
nan

np.nanmean(y_with_nan)

mean_ = z.mean()
mean_

z_with_nan.mean()

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean

wmean = np.average(z, weights=w)
wmean

(w * y).sum() / w.sum()

w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)

hmean = len(x) / sum(1 / item for item in x)
hmean

hmean = statistics.harmonic_mean(x)
hmean

statistics.harmonic_mean(x_with_nan)
statistics.harmonic_mean([1, 0, 2])
statistics.harmonic_mean([1, 2, -2])

scipy.stats.hmean(y)

scipy.stats.hmean(z)

gmean = 1
    for item in x:
        gmean *= item

gmean **= 1 / len(x)
    gmean

gmean = statistics.geometric_mean(x)
gmean 

gmean = statistics.geometric_mean(x_with_nan)
gmean 

scipy.stats.gmean(y)
scipy.stats.gmean(z)

n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])

 median_

median_ = statistics.median(x)
median_

median_ = statistics.median(x[:-1])
median_

statistics.median_low(x[:-1])
statistics.median_high(x[:-1])

statistics.median(x_with_nan)
statistics.median_low(x_with_nan)
statistics.median_high(x_with_nan)

median_ = np.median(y)
median_

median_ = np.median(y[:-1])
median_

np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])

z.median()
z_with_nan.median()

u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

mode_ = statistics.mode(u)
mode_ 

v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v) 
statistics.multimode(v)

statistics.mode([2, math.nan, 2])
statistics.multimode([2, math.nan, 2])
statistics.mode([2, math.nan, 0, math.nan, 5])
statistics.multimode([2, math.nan, 0, math.nan, 5])

u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_

mode_ = scipy.stats.mode(v)
mode_

mode_.mode

mode_.count

u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
v.mode()
w.mode()

n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

var_ = statistics.variance(x)
var_

statistics.variance(x_with_nan)

var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_

np.nanvar(y_with_nan, ddof=1)

z.var(ddof=1)
z_with_nan.var(ddof=1)

std_ = var_ ** 0.5
std_ 

np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)
y_with_nan.std(ddof=1)
np.nanstd(y_with_nan, ddof=1)

z.std(ddof=1)

z_with_nan.std(ddof=1)

x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
        * n / ((n - 1) * (n - 2) * std_**3))
skew_

y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
1.9470432273905927
scipy.stats.skew(y_with_nan, bias=False)

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()

z_with_nan.skew()

x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive')

y = np.array(x)
np.percentile(y, 5)

np.percentile(y, 95)

np.percentile(y, [25, 50, 75])
np.median(y)

y_with_nan = np.insert(y, 2, np.nan)
y_with_nan

np.nanpercentile(y_with_nan, [25, 50, 75])

np.quantile(y, 0.05)

np.quantile(y, 0.95)

np.quantile(y, [0.25, 0.5, 0.75])

np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)

z.quantile(0.95)

z.quantile([0.25, 0.5, 0.75])

z_with_nan.quantile([0.25, 0.5, 0.75])

np.ptp(y)
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()

quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

result = scipy.stats.describe(y, ddof=1, bias=False)
result 

result.nobs
result.minmax[0]
result.minmax[1]
result.mean
result.variance
result.skewness
result.kurtosis

result = z.describe()
result

result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

cov_matrix = np.cov(x_, y_)
cov_matrix

x_.var(ddof=1)
y_.var(ddof=1)

cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix[1, 0]
cov_xy

cov_xy = x__.cov(y__)
cov_xy

var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

r, p = scipy.stats.pearsonr(x_, y_)
r 
p

corr_matrix = np.corrcoef(x_, y_)
corr_matrix

scipy.stats.linregress(x_, y_)

result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r