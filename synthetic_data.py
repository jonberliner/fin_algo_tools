import numpy as np
from scipy.optimize import minimize

clip, exp, sin, randn = np.clip, np.exp, np.sin, np.random.randn

def sin_data(num, x0=0., x1=np.pi*2., amp=1., freq=1., phase=0., length=128, seed=None):
    rng = np.random.RandomState(seed)
    def rand(lo, hi):
        return rng.uniform(lo, hi)
    xs = np.linspace(x0, x1, length)
    out = list()
    for ii in range(num):
        ph = rand(*phase) if type(phase) is tuple else phase
        fr = rand(*freq) if type(freq) is tuple else freq
        am = rand(*amp) if type(amp) is tuple else amp
        out.append(am * np.sin(fr * xs + ph))
    return np.stack(out)


clip, exp, sin, randn = np.clip, np.exp, np.sin, np.random.randn

def sigmoid(x, fudge=1e-6):
    return clip(1. / (1. + exp(-x)), fudge, 1.-fudge)

# def damp_sin(xs, amp, freq, phase, damp, eps, seed=None):
#     ys = ((sigmoid(damp)**xs) * exp(amp))\
#          * sin(exp(freq) * xs + phase)\
#          + randn(*xs.shape) * exp(eps)
#     return ys

def damp_sin(xs, amp, freq, phase, damp, eps, seed=None):
    ys = ((sigmoid(damp)**xs) * amp)\
         * sin(freq * xs + phase)\
         + randn(*xs.shape) * eps
    return ys

def build_loss_fn(xs, ys):
    def fn(params):
        amp, freq, phase, damp, eps = params
        yh = damp_sin(xs, amp, freq, phase, damp, eps)
        return np.sum(np.square(yh - ys))
    return fn

SEED = None
rng = np.random.RandomState(SEED)
length = 128
subsample = 40

xs = np.linspace(0, np.pi*2, length)

real_params = [4.0, 6.0, 0.0, 1., 0.2]
ys = damp_sin(xs, *real_params)

# i0 = np.sort(rng.choice(length, subsample, replace=False))
i0 = np.arange(subsample)
x0 = xs[i0]
y0 = ys[i0]

loss_fn = build_loss_fn(x0, y0)
fudge = 1e-6
init_params = [2., 4., fudge, 0., fudge]
# bounds = ((fudge, None), (fudge, None), (0., 0.), (fudge, 1. - fudge), (fudge, fudge*3))
# bounds = ((fudge, None), (0.5, None), (0., 0.), (1., 1.), (0., 0.))
# bounds = ((fudge, 1000.), (fudge, 100.), (0., np.pi*2.), (fudge, 1.-fudge), (fudge, fudge))

res = basinhopping(loss_fn, init_params, disp=True, niter=100)

# bounds = ((fudge, None), (fudge, None), (0., np.pi*2.), (None, None), (fudge, None))
# res = minimize(loss_fn, init_params, bounds=bounds, method='SLSQP', tol=1e-12)

yh = damp_sin(xs, *res.x)



class DampSinFitter(object):
    DAMP_SIN_KEYS = ['amp', 'freq', 'phase', 'damp', 'eps']
    def __init__(self, bounds=None, fudge=1e-6):
        self.fudge=fudge
        self.bounds_dict = {'amp': (self.fudge, None),
                            'freq': (self.fudge, None),
                            'phase': (None, None),
                            'damp': (self.fudge, 1.-self.fudge),
                            'eps': (self.fudge, None)}
        self.bounds = [self.bounds_dict[key] for key in DampSinFitter.DAMP_SIN_KEYS]
        # NOTE: have found that non-global optimization struggles a lot with this problem
        self.optimizer = basinhopping

    def fit(self, xs, ys, **opt_kwargs):
        loss_fn = self.build_loss_fn(xs, ys)
        res = self.optimizer(loss_fn, self.bounds, **opt_kwargs)


    def sigmoid(x, fudge=1e-6):
        return clip(1. / (1. + exp(-x)), fudge, 1.-fudge)

    # def damp_sin(xs, amp, freq, phase, damp, eps, seed=None):
    #     ys = ((sigmoid(damp)**xs) * exp(amp))\
    #          * sin(exp(freq) * xs + phase)\
    #          + randn(*xs.shape) * exp(eps)
    #     return ys

    def damp_sin(xs, amp, freq, phase, damp, eps, seed=None):
        ys = ((sigmoid(damp)**xs) * amp)\
            * sin(freq * xs + phase)\
            + randn(*xs.shape) * eps
        return ys

    def build_loss_fn(xs, ys):
        def fn(params):
            amp, freq, phase, damp, eps = params
            yh = damp_sin(xs, amp, freq, phase, damp, eps)
            return np.sum(np.square(yh - ys))
        return fn



# res = minimize(loss_fn, init_params, bounds=bounds, tol=1e-12, method='basin_hopping', options=dict(maxiter=int(1e4), disp=True))



# def dampening_sin_data(num, x0=0., x1=np.pi*2., amp=1., freq=1., phase=0., length=128, damp=1., seed=None):
#     rng = np.random.RandomState(seed)
#     def rand(lo, hi):
#         return rng.uniform(lo, hi)
#     xs = np.linspace(x0, x1, length)
#     out = list()
#     for ii in range(num):
#         ph = rand(*phase) if type(phase) is tuple else phase
#         fr = rand(*freq) if type(freq) is tuple else freq
#         am = rand(*amp) if type(amp) is tuple else amp
#         dam = rand(*damp) if type(damp) is tuple else damp
#         dam = dam ** xs
#         out.append((dam * am) * np.sin(fr * xs + ph))
#     return np.stack(out)
