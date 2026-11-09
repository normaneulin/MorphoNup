"""Keras layers of the MorphoNup graph.

Class names must stay as they are: Keras 3 `.weights.h5` files key weights by snake-cased class name
and graph order, so renaming a class breaks loading saved models.
"""
import numpy as np
import tensorflow as tf

L = tf.keras.layers
register = tf.keras.utils.register_keras_serializable


@register(package="MorphoNup")
class GroupedJointMotifs(L.Layer):
    """Sequence PWMs attenuated by RBF distance to physical prototypes, one bandwidth per group.

    Analysis arms switch off one term: sequence=False keeps only the shape-prototype similarity (shape-only
    motifs); physical=False keeps only the PWM match (plain sequence motifs). Defaults = MorphoNup.
    """

    def __init__(self, k, count, groups, sequence=True, physical=True, **kwargs):
        super().__init__(**kwargs)
        self.k, self.count = int(k), int(count)
        self.groups = [list(map(int, g)) for g in groups]
        self.channels = sum(len(g) for g in self.groups)
        self.sequence, self.physical = bool(sequence), bool(physical)
        if not (self.sequence or self.physical):
            raise ValueError("A motif needs a sequence or a physical term.")

    def build(self, shapes):
        if self.sequence:
            self.motif_logits = self.add_weight(name="motif_logits", shape=(self.count, self.k, 4),
                                                initializer=tf.keras.initializers.RandomNormal(stddev=0.2))
        if self.physical:
            self.physical_centers = self.add_weight(name="physical_centers", shape=(self.count, 3, self.channels),
                                                    initializer="zeros")
            self.bandwidth_logits = self.add_weight(name="bandwidth_logits", shape=(len(self.groups),),
                                                    initializer=tf.keras.initializers.Constant(np.log(1 / 2.5)))
            self.strength_logit = self.add_weight(name="strength_logit", shape=(),
                                                  initializer=tf.keras.initializers.Constant(np.log(0.1 / 0.9)))
        super().build(shapes)

    def call(self, inputs):
        sequence, physical = inputs
        if self.sequence:
            motif = tf.math.l2_normalize(tf.nn.softmax(self.motif_logits, axis=-1), axis=-1)
            score = tf.nn.conv1d(sequence, tf.transpose(motif, [1, 2, 0]), 1, "VALID")
            matches = tf.exp(0.7 * (score - float(self.k)))
            if not self.physical:
                return matches
        crop = self.k // 2 - 3
        track = physical[:, crop:tf.shape(physical)[1] - crop] if crop else physical
        patch = tf.signal.frame(track, 3, 1, axis=1)
        widths = 0.5 + 3.5 * tf.sigmoid(self.bandwidth_logits)
        distances = []
        for i, indices in enumerate(self.groups):
            x = tf.gather(patch, indices, axis=-1)
            c = tf.gather(self.physical_centers, indices, axis=-1)
            d = (tf.reduce_sum(x * x, axis=[2, 3])[:, :, None] + tf.reduce_sum(c * c, axis=[1, 2])[None, None, :]
                 - 2.0 * tf.einsum("bltc,mtc->blm", x, c)) / (3.0 * len(indices))
            distances.append(tf.maximum(d, 0.0) / widths[i] ** 2)
        attenuation = tf.exp(-tf.sigmoid(self.strength_logit) * tf.add_n(distances) / len(self.groups))
        return matches * attenuation if self.sequence else attenuation

    def compute_output_shape(self, shapes):
        return (shapes[0][0], 148 - self.k, self.count)

    def get_config(self):
        return {**super().get_config(), "k": self.k, "count": self.count, "groups": self.groups,
                "sequence": self.sequence, "physical": self.physical}


@register(package="MorphoNup")
class DyadPool(L.Layer):
    """Global mean/max and seven soft spatial bins, without learned attention."""
    def call(self,x):
        pos=tf.linspace(-1.,1.,tf.shape(x)[1]);centers=tf.linspace(-1.,1.,7)
        weights=tf.exp(-.5*((pos[:,None]-centers[None,:])/.25)**2)
        weights/=tf.reduce_sum(weights,axis=0,keepdims=True)
        bins=tf.einsum('blc,lp->bpc',x,weights)
        return tf.concat([tf.reduce_mean(x,axis=1),tf.reduce_max(x,axis=1),
            tf.reshape(bins,[tf.shape(x)[0],7*int(x.shape[-1])])],axis=1)

    def compute_output_shape(self,shape):return (shape[0],9*shape[-1])


@register(package="MorphoNup")
class GlobalPool(L.Layer):
    """Global mean and max only (DyadPool without its seven dyad bins; analysis arm)."""
    def call(self, x):
        return tf.concat([tf.reduce_mean(x, axis=1), tf.reduce_max(x, axis=1)], axis=1)

    def compute_output_shape(self, shape):
        return (shape[0], 2 * shape[-1])


@register(package="MorphoNup")
class CenterCrop1D(L.Layer):
    """Keeps the central `length` positions (motif maps of different widths share one centre; candidate arm only)."""
    def __init__(self, length, **kw):
        super().__init__(**kw)
        self.length = length

    def call(self, x):
        start = (tf.shape(x)[1] - self.length) // 2
        return x[:, start:start + self.length]

    def compute_output_shape(self, shape):
        return (shape[0], self.length, shape[-1])

    def get_config(self):
        return {**super().get_config(), "length": self.length}


@register(package="MorphoNup")
class ZeroConnect(L.Layer):
    """Adds 0 x inputs an analysis arm does not use, so every arm keeps the same three-input contract."""
    def call(self, inputs):
        out, *unused = inputs
        return out + tf.add_n([0.0 * tf.reduce_sum(u, axis=list(range(1, len(u.shape))))[:, None] for u in unused])

    def compute_output_shape(self, shapes):
        return shapes[0]


@register(package="MorphoNup")
class MotifSpectrum(L.Layer):
    """61-bp quadrature filters on motif activations; shared periods/widths.

    Periods 8..14 bp, widths 8..20 bp; paired filters use a common normalization.
    Reflection padding and zero DC response. No phase parameter is needed for
    quadrature energy. This tests recurrence of learned motifs, not image texture.
    """
    def build(self,shape):
        periods=np.array([9.5,10.2,10.8,11.5],np.float32)
        v=(periods-8.)/6.
        self.period_logits=self.add_weight(name='period_logits',shape=(4,),
            initializer=tf.keras.initializers.Constant(np.log(v/(1-v))))
        self.width_logits=self.add_weight(name='width_logits',shape=(4,),initializer='zeros')
        self.channels=int(shape[-1]);super().build(shape)

    def parameters(self):return 8.+6.*tf.sigmoid(self.period_logits),8.+12.*tf.sigmoid(self.width_logits)

    def call(self,x):
        period,width=self.parameters();t=tf.range(-30,31,dtype=x.dtype)[:,None]
        envelope=tf.exp(-.5*(t/width[None,:])**2)
        angle=2*np.pi*t/period[None,:]
        kernel=tf.stack([envelope*tf.cos(angle),envelope*tf.sin(angle)],axis=-1)
        kernel-=tf.reduce_mean(kernel,axis=0,keepdims=True)
        kernel/=tf.sqrt(tf.reduce_sum(kernel**2,axis=[0,2],keepdims=True)+1e-8)
        kernel=tf.tile(tf.reshape(kernel,[1,61,1,8]),[1,1,self.channels,1])
        padded=tf.pad(x,[[0,0],[30,30],[0,0]],mode='REFLECT')
        response=tf.nn.depthwise_conv2d(padded[:,None],kernel,[1,1,1,1],'VALID')
        response=tf.reshape(response,[tf.shape(x)[0],tf.shape(x)[1],self.channels,4,2])
        energy=tf.math.log1p(tf.reduce_sum(response**2,axis=-1))
        return tf.reshape(tf.concat([tf.reduce_mean(energy,1),tf.reduce_max(energy,1)],axis=-1),
                          [tf.shape(x)[0],self.channels*8])

    def compute_output_shape(self,shape):return (shape[0],shape[-1]*8)


@register(package="MorphoNup")
class PhysicalMoments(L.Layer):
    def call(self,x):
        return tf.concat([tf.reduce_mean(x,axis=1),tf.math.reduce_std(x,axis=1)],axis=-1)
    def compute_output_shape(self,shape):return (shape[0],2*shape[-1])


@register(package="MorphoNup")
class ConnectedAnchor(L.Layer):
    def call(self,inputs):
        z,h,p=inputs
        # Keep the same input contract even in the one-parameter bias control.
        return z+0.*tf.reduce_sum(h,axis=1,keepdims=True)+0.*tf.reduce_sum(p,axis=[1,2])[:,None]


@register(package="MorphoNup")
class BoundedCorrection(L.Layer):
    def __init__(self,limit=1.,penalty=.02,**kwargs):
        super().__init__(**kwargs);self.limit=float(limit);self.penalty=float(penalty)
    def build(self,shapes):
        self.gate=self.add_weight(name='gate',shape=(),initializer='ones',trainable=False)
        super().build(shapes)
    def call(self,inputs):
        anchor,delta=inputs;delta=self.limit*tf.tanh(delta/self.limit)
        self.add_loss(self.penalty*tf.reduce_mean(delta**2))
        return anchor+self.gate*delta
    def get_config(self):return {**super().get_config(),'limit':self.limit,'penalty':self.penalty}


@register(package="MorphoNup")
class ChannelSubset(L.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs); self.indices = list(indices)

    def call(self, x):
        return tf.gather(x, self.indices, axis=-1)

    def get_config(self):
        return {**super().get_config(), "indices": self.indices}


def nucleotide_words(x,k):
    """Differentiable A,C,G,T lexical word channels, not an argmax encoding."""
    length=tf.shape(x)[1]-k+1
    result=x[:,:length]
    for j in range(1,k):
        result=tf.reshape(result[:,:,:,None]*x[:,j:j+length,None,:],[tf.shape(x)[0],length,4**(j+1)])
    return result


@register(package="MorphoNup")
class MotifStatistics(L.Layer):
    """80 dimer/trimer identities: 7 dyad bins + 3 composition-adjusted recurrence features."""
    def call(self,x):
        pieces=[]
        for k in (2,3):
            hits=nucleotide_words(x,k);length=148-k
            pos=np.arange(length)+(k-1)/2.-73.
            centers=np.linspace(-63.,63.,7)
            weights=np.exp(-.5*((pos[:,None]-centers[None,:])/12.)**2)
            weights/=weights.sum(axis=0,keepdims=True)
            bins=tf.einsum('blc,lj->bjc',hits,tf.constant(weights,dtype=hits.dtype))
            pieces.append(tf.reshape(bins,[tf.shape(x)[0],7*4**k]))
            for period in (10,11,12):
                correlations=[]
                for multiple in (1,2,3,4):
                    lag=period*multiple;a,b=hits[:,:-lag],hits[:,lag:]
                    ma,mb=tf.reduce_mean(a,1),tf.reduce_mean(b,1)
                    cov=tf.reduce_mean(a*b,1)-ma*mb
                    variance=ma*(1-ma)*mb*(1-mb)
                    correlations.append(cov/tf.sqrt(variance+1e-6))
                pieces.append(tf.add_n(correlations)/4.)
        return tf.concat(pieces,axis=-1)
    def compute_output_shape(self,shape):return (shape[0],800)


@register(package="MorphoNup")
class StatisticsByWord(L.Layer):
    """Reorder the existing 800 statistics into 80 motifs x 10 descriptors."""
    def call(self,x):
        pieces=[];offset=0
        for count in (16,64):
            values=tf.reshape(x[:,offset:offset+10*count],[tf.shape(x)[0],10,count])
            pieces.append(tf.transpose(values,[0,2,1]));offset+=10*count
        return tf.concat(pieces,axis=1)
    def compute_output_shape(self,shape):return (shape[0],80,10)


def symmetric_bernoulli_kl(p,q):
    p=tf.clip_by_value(p,1e-6,1.-1e-6);q=tf.clip_by_value(q,1e-6,1.-1e-6)
    return .5*(p-q)*(tf.math.log(p)-tf.math.log1p(-p)-tf.math.log(q)+tf.math.log1p(-q))


def bernoulli_js(a, b):
    """Per-record Jensen-Shannon divergence between Bernoulli probabilities."""
    a, b = tf.clip_by_value(a, 1e-6, 1-1e-6), tf.clip_by_value(b, 1e-6, 1-1e-6)
    m = (a+b)/2
    def kl(p):
        return p*tf.math.log(p/m)+(1-p)*tf.math.log((1-p)/(1-m))
    return .5*(kl(a)+kl(b))
