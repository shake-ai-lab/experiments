"""俗説「ReLUはsigmoidの簡略化(区分線形近似)である」を、実際にカーブフィットして検証する。

sigmoid(x) に対して、scale*ReLU(x + shift) + offset の形の関数を最小二乗でフィットさせ、
どれだけ良く/悪く近似できるかをMSEとR^2で定量化する。
"""
import numpy as np
from scipy.optimize import curve_fit

OUTDIR = "C:/tmp/activation-function-myths"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def scaled_relu(x, scale, shift, offset):
    return scale * relu(x + shift) + offset


xs = np.linspace(-8, 8, 2000)
ys_sigmoid = sigmoid(xs)

popt, _ = curve_fit(scaled_relu, xs, ys_sigmoid, p0=[0.25, 0, 0], maxfev=20000)
scale, shift, offset = popt
ys_fit = scaled_relu(xs, *popt)

mse = np.mean((ys_sigmoid - ys_fit) ** 2)
ss_res = np.sum((ys_sigmoid - ys_fit) ** 2)
ss_tot = np.sum((ys_sigmoid - ys_sigmoid.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
max_abs_err = np.max(np.abs(ys_sigmoid - ys_fit))

print(f"best-fit scaled ReLU: scale={scale:.4f} shift={shift:.4f} offset={offset:.4f}")
print(f"MSE={mse:.5f}  R^2={r2:.4f}  max_abs_error={max_abs_err:.4f}")

# also compare against a "fair" baseline: how well does a horizontal line (mean) fit? (R^2=0 baseline)
# and how well would a smooth alternative (e.g. tanh-based logistic itself) fit trivially (R^2=1, sanity check)
mean_baseline_mse = np.mean((ys_sigmoid - ys_sigmoid.mean()) ** 2)

# behavior comparison at extremes: does ReLU saturate like sigmoid?
x_extreme = np.array([-8, -4, -2, 0, 2, 4, 8])
sig_vals = sigmoid(x_extreme)
relu_vals = relu(x_extreme)
print("\nx:          ", x_extreme)
print("sigmoid(x): ", np.round(sig_vals, 4))
print("relu(x):    ", np.round(relu_vals, 4))

# extrapolation test: same fitted scaled-ReLU, evaluated far outside the fitting window [-8,8]
xs_wide = np.linspace(-30, 30, 4000)
ys_sigmoid_wide = sigmoid(xs_wide)
ys_fit_wide = scaled_relu(xs_wide, *popt)
max_abs_err_wide = np.max(np.abs(ys_sigmoid_wide - ys_fit_wide))
at_30 = dict(sigmoid=float(sigmoid(30)), fit=float(scaled_relu(30, *popt)))
print(f"\nextrapolation to x in [-30,30]: max_abs_error={max_abs_err_wide:.2f}")
print(f"at x=30: sigmoid={at_30['sigmoid']:.4f} (saturates near 1)  "
      f"best-fit scaled ReLU={at_30['fit']:.2f} (keeps growing unbounded)")

np.savez(
    f"{OUTDIR}/curve_fit_results.npz",
    xs=xs, ys_sigmoid=ys_sigmoid, ys_fit=ys_fit,
    scale=scale, shift=shift, offset=offset,
    mse=mse, r2=r2, max_abs_err=max_abs_err, mean_baseline_mse=mean_baseline_mse,
    x_extreme=x_extreme, sig_vals=sig_vals, relu_vals=relu_vals,
    xs_wide=xs_wide, ys_sigmoid_wide=ys_sigmoid_wide, ys_fit_wide=ys_fit_wide,
    max_abs_err_wide=max_abs_err_wide, at_30_sigmoid=at_30["sigmoid"], at_30_fit=at_30["fit"],
)
print("\nsaved curve_fit_results.npz")
