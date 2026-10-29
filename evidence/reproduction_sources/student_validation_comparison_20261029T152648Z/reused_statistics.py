import numpy as np
import math

class Stats:
    def __init__(self):
        self.n = self.floor = self.over2 = self.under2 = self.pred_far = 0
        self.err = self.ae = self.sse = self.ar = self.rel_bias = self.pred = self.gt = 0.
        self.deltas = np.zeros(3, dtype=np.int64)

    def add(self, p, g):
        assert p.dtype == g.dtype == np.float64
        if not g.size:
            return
        err = p - g
        ratio = np.maximum(p / g, g / p)
        self.n += len(g)
        self.floor += int(np.count_nonzero(p <= .101))
        self.over2 += int(np.count_nonzero(err >= 2.))
        self.under2 += int(np.count_nonzero(err <= -2.))
        self.pred_far += int(np.count_nonzero(p >= 10.))
        self.err += float(err.sum())
        self.ae += float(np.abs(err).sum())
        self.sse += float(np.square(err).sum())
        self.ar += float((np.abs(err) / g).sum())
        self.rel_bias += float((err / g).sum())
        self.pred += float(p.sum())
        self.gt += float(g.sum())
        self.deltas += [np.count_nonzero(ratio < 1.25**k) for k in (1, 2, 3)]

    def result(self):
        if not self.n:
            return {'valid_pixels': 0, 'absrel': None, 'rmse': None, 'bias_m': None}
        return dict(valid_pixels=self.n, absrel=self.ar / self.n, rmse=math.sqrt(self.sse / self.n),
                    mae_m=self.ae / self.n, bias_m=self.err / self.n,
                    mean_signed_relative_error=self.rel_bias / self.n,
                    mean_prediction_m=self.pred / self.n, mean_gt_m=self.gt / self.n,
                    floor_count=self.floor, floor_rate=self.floor / self.n,
                    overestimate_ge_2m_rate=self.over2 / self.n,
                    underestimate_ge_2m_rate=self.under2 / self.n,
                    prediction_ge_10m_rate=self.pred_far / self.n,
                    delta1=self.deltas[0] / self.n, delta2=self.deltas[1] / self.n,
                    delta3=self.deltas[2] / self.n)
