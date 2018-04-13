from __future__ import print_function, division
import numpy as np
from scipy.optimize import basinhopping

# pull out of dots for speed
npmean, clip, exp, sin, randn, npsum, npsquare, log = np.mean, np.clip, np.exp, np.sin, np.random.randn, np.sum, np.square, np.log

def sigmoid(x, fudge=1e-6):
    return clip(1. / (1. + exp(-x)), fudge, 1.-fudge)


def _softplus(elt, limit=30., fudge=1e-6):
    """softplus fn w stability. when elt gets large, log(1. + exp(x)) ~= x.
    we thus sidestep overflow worries for exp(x)"""
    if elt < limit:
        return log(1. + exp(x)) + fudge
    else:
        return elt + fudge

softplus = np.vectorize(_softplus)


# experimenting with softplus or exp.  softplus normally better for neural approaches,
# but exp seems to outperform here
log_transform = exp
# log_transform = softplus


def damp_sin(xs, log_amp, log_freq, phase, logit_damp):
    """build a dampening sine-wave with values at locations xs"""
    # vals transformed bc cannot put bounds on basinhopping
    ys = ((sigmoid(logit_damp)**xs) * log_transform(log_amp))\
            * sin(log_transform(log_freq) * xs + phase)
            # + randn(*xs.shape) * log_transform(log_eps)
    return ys


def add_white_noise(ys, log_var, n=None):
    """add white noise with log variance log_var to data ys
    n times.  if n is None, does not add sampling dimension"""
    _n = n or 1
    noise = randn(_n, len(ys)) * exp(log_var)
    out = ys + noise
    return np.squeeze(out, 0) if not n else out


class DampSinFitter(object):
    """fits data only assuming it's searching for dampening sine waves"""
    DAMP_SIN_KEYS = ['log_amp', 'log_freq', 'phase', 'logit_damp', 'noise_lv']
    def __init__(self, init_params, fudge=1e-6):
        # NOTE: have found that non-global optimization struggles a lot with this problem
        self.optimizer = basinhopping  # our global optimizer
        self.damp_sin = damp_sin  # dampening sin function with transformed params so can optimize in the real domain

        self.fudge=fudge  # stay away from extreme values

        # NOTE: be careful - looks like need some sort of reasonable initialization to get good fits

        # readable dict
        self.init_params = init_params
        # non-readable vector for optimizer
        self._init_params = [self.init_params[key] for key in DampSinFitter.DAMP_SIN_KEYS]

        self._set_params(self._init_params)

    def _set_params(self, _params):
        # non-readable format for optimizer
        self._params = _params
        # readable
        self.params = {key: val for key, val in zip(DampSinFitter.DAMP_SIN_KEYS, _params)}

    def fit(self, xs, ys, nsam=1, **opt_kwargs):
        """fit our damp sin params given observations ys at locations xs.
        nsam will sample with noise var exp(noise_lv) nsam times per update"""
        loss_fn = self.build_loss_fn(xs, ys, nsam)
        res = self.optimizer(loss_fn, self._params, **opt_kwargs)
        self._set_params(res.x)
        return res

    def predict(self, xs, nsam=None):
        """if nsam passed, return nsam noisy samples.  else return noiseless mean"""
        mu = out = self.damp_sin(xs, *self._params[:-1])
        if nsam:
            out = add_white_noise(mu, self.params['noise_lv'], nsam)
        return out

    def score(self, yhat, ytrue):
        """function for the optimizer to minimize"""
        return npmean(npsquare(yhat - ytrue))

    def build_loss_fn(self, xs, ys, nsam):
        """take in training data xs and ys and return a loss function for 
        getting MLE values for damp sine params.  nsam will use nsam noisy
        samples of yhat per update"""
        score, damp_sin = self.score, self.damp_sin
        def fn(params):
            lamp, lfreq, phase, ldamp, noise_lv = params
            clean = damp_sin(xs, lamp, lfreq, phase, ldamp)
            yh = add_white_noise(clean, noise_lv, nsam)
            return self.score(yh, ys)
        return fn


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    SEED = 3355
    rng = np.random.RandomState(SEED)
    LENGTH = 256
    SUBSAMPLE = 64


    REAL_PARAMS = dict(log_amp=5.,
                       log_freq=2.,
                       phase=0.,
                       logit_damp=1.,
                       noise_lv=3.)
    _real_params = [REAL_PARAMS[key] for key in DampSinFitter.DAMP_SIN_KEYS]

    # GENERATE SOME "TRUE" DATA
    xs = np.linspace(0, np.pi*2, LENGTH)
    ys_clean = damp_sin(xs, *_real_params[:-1])

    # gather a few "noisy" samples to train over (first n steps of time-series)
    ys = add_white_noise(ys_clean, _real_params[-1])
    i0 = np.arange(SUBSAMPLE)
    x0 = xs[i0]
    y0 = ys[i0]

    # BUILD OUR MODEL
    # initial guess for our model to start with
    INIT_PARAMS = dict(log_amp=8.,
                       log_freq=1.,
                       phase=2.,
                       logit_damp=0.,
                       noise_lv=-0.2)
    fitter = DampSinFitter(init_params=INIT_PARAMS)

    # fit sine to data we have 
    # NOTE: may not always work!  rerun if doesn't. would prefer something more
    #       robust that don't have to worry, but so it goes
    fitter.fit(x0, y0, nsam=10, niter=300)

    # get params for inspection if you want
    fit_params = fitter.params

    # extract predictions
    yh = fitter.predict(xs)

    FIGNAME = 'damp_sin_fitter_example.png'
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys_clean, 'k', label='true signal')
    ax.plot(x0, y0, 'ko', label='training data')
    ax.plot(xs, yh, 'r', label='inferred signal')
    ax.set_title('fitted dampening sin wave')
    plt.legend()
    plt.show()
    # print('saving fit to {:s}'.format(FIGNAME))
    # plt.savefig(FIGNAME, bbox_inches='tight')
