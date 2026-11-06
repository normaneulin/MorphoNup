"""Training of the MorphoNup core: SAM over all weights + AdamW, tuning-only checkpoint selection, refit.

Per batch of n records the core sees 4n inputs:
[forward, RC, forward, RC], i.e. two dropout passes of both strands. Objective =
mean BCE over the 4 passes + rc_consistency_weight * JS(forward, RC) within each pass
+ rdrop_weight * symmetric KL between the two passes' strand-averaged probabilities
+ the bounded-correction penalties (core.losses).
Analysis arms only: cfg["rc_view"] = False trains and predicts on the forward strand alone (two dropout passes,
no JS term); cfg["sam"] = False takes plain AdamW steps. Both default to True (MorphoNup).
"""
import numpy as np
import tensorflow as tf

from .layers import bernoulli_js, symmetric_bernoulli_kl
from .metrics import scores


def state_variables(core):
    # Include RNG states explicitly: backend versions differ in which non-weight
    # variables appear in non_trainable_variables. De-duplicate by object identity.
    values = list(core.non_trainable_variables)
    for layer in core._flatten_layers():
        generator = getattr(layer, "seed_generator", None)
        if generator is not None:
            values.append(generator.state)
    return list({id(v): v for v in values}.values())


def perturbations(gradients, rho):
    norm = tf.linalg.global_norm(gradients)
    factor = tf.cast(rho, norm.dtype) / (norm + 1e-12)
    return [g * factor for g in gradients], norm


class Trainer(tf.keras.Model):
    """Wraps the core (attribute name `core` is part of the saved-weights layout)."""

    def __init__(self, cfg, core):
        super().__init__(name="MorphoNup_training")
        self.cfg, self.core = cfg, core
        self.trackers = {n: tf.keras.metrics.Mean(name=n) for n in ("loss", "bce", "rc_js", "dropout_kl")}
        self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.auc_tracker = tf.keras.metrics.AUC(name="auc")
        self.bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg["label_smoothing"], reduction="none")
        self.sam_rho = cfg["sam_rho"]
        self.sam_loss_tracker = tf.keras.metrics.Mean(name="sam_loss")
        self.sam_delta_tracker = tf.keras.metrics.Mean(name="sam_delta")
        self.sam_norm_tracker = tf.keras.metrics.Mean(name="sam_gradient_norm")

    @property
    def metrics(self):
        return [*self.trackers.values(), self.accuracy_tracker, self.auc_tracker,
                self.sam_loss_tracker, self.sam_delta_tracker, self.sam_norm_tracker]

    def call(self, inputs, training=False):
        return tf.sigmoid(self.core(inputs, training=training))

    def objective(self, inputs, y, training=False):
        a, b = inputs
        if not self.cfg.get("rc_view", True):
            joined = {k: tf.concat([a[k], a[k]], axis=0) for k in a}
            p, q = tf.split(self(joined, training=training), 2, axis=0)              # two passes, forward only
            y = tf.cast(tf.reshape(y, [-1, 1]), tf.float32)
            bce = 0.5 * (self.bce(y, p) + self.bce(y, q))
            js = tf.zeros_like(tf.reshape(p, [-1]))
            kl = tf.reshape(symmetric_bernoulli_kl(p, q), [-1])
            return self._total(bce, js, kl, y, p, q)
        joined = {k: tf.concat([a[k], b[k], a[k], b[k]], axis=0) for k in a}
        f1, r1, f2, r2 = tf.split(self(joined, training=training), 4, axis=0)   # pass 1 fwd/RC, pass 2 fwd/RC
        p, q = 0.5 * (f1 + r1), 0.5 * (f2 + r2)
        y = tf.cast(tf.reshape(y, [-1, 1]), tf.float32)
        bce = 0.25 * (self.bce(y, f1) + self.bce(y, r1) + self.bce(y, f2) + self.bce(y, r2))
        js = tf.reshape(0.5 * (bernoulli_js(f1, r1) + bernoulli_js(f2, r2)), [-1])
        kl = tf.reshape(symmetric_bernoulli_kl(p, q), [-1])
        return self._total(bce, js, kl, y, p, q)

    def _total(self, bce, js, kl, y, p, q):
        bce, js, kl = map(tf.reduce_mean, (bce, js, kl))
        penalty = tf.add_n(self.core.losses) if self.core.losses else tf.constant(0.0, tf.float32)
        loss = bce + self.cfg["rc_consistency_weight"] * js + self.cfg["rdrop_weight"] * kl + penalty
        return loss, bce, js, y, 0.5 * (p + q), kl

    def update(self, result):
        loss, bce, js, y, p, kl = result
        for name, value in zip(("loss", "bce", "rc_js", "dropout_kl"), (loss, bce, js, kl)):
            self.trackers[name].update_state(value)
        self.accuracy_tracker.update_state(y, p)
        self.auc_tracker.update_state(y, p)
        return {t.name: t.result() for t in self.metrics}

    def sam_gradients(self, inputs, labels):
        """Two loss/gradient evaluations, no optimizer step; exact state restoration.

        The same RNG realization is used at original and perturbed weights.
        BN still uses batch statistics in both passes, but only the first pass's
        running-stat updates persist. Random generators advance once per batch.
        """
        variables = self.trainable_variables
        state = state_variables(self.core)
        before = [tf.identity(v) for v in state]
        with tf.GradientTape() as tape:
            first = self.objective(inputs, labels, True)
        gradient = tape.gradient(first[0], variables)
        assert all(g is not None for g in gradient), "Disconnected first-pass gradient"
        gradient = [tf.debugging.check_numerics(tf.convert_to_tensor(g), "SAM first gradient") for g in gradient]
        after = [tf.identity(v) for v in state]
        original = [tf.identity(v) for v in variables]
        epsilon, norm = perturbations(gradient, self.sam_rho)
        try:
            for v, e in zip(variables, epsilon):
                v.assign_add(e)
            for v, value in zip(state, before):
                v.assign(value)
            with tf.GradientTape() as tape:
                second = self.objective(inputs, labels, True)
            adjusted = tape.gradient(second[0], variables)
            assert all(g is not None for g in adjusted), "Disconnected second-pass gradient"
            adjusted = [tf.identity(tf.convert_to_tensor(g)) for g in adjusted]
        finally:
            # Eager exceptions restore weights too; graph execution restores after
            # all second-pass gradient tensors have materialized.
            dependencies = adjusted if "adjusted" in locals() else []
            with tf.control_dependencies(dependencies):
                restore = [v.assign(value) for v, value in zip(variables, original)]
                restore.extend(v.assign(value) for v, value in zip(state, after))
        with tf.control_dependencies(restore):
            adjusted = [tf.debugging.check_numerics(g, "SAM second gradient") for g in adjusted]
            second_loss = tf.identity(second[0])
        return first, adjusted, second_loss, norm

    def train_step(self, data):
        inputs, labels, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        if not self.cfg.get("sam", True):
            with tf.GradientTape() as tape:
                first = self.objective(inputs, labels, True)
            gradient = tape.gradient(first[0], self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
            return self.update(first)
        first, gradient, second_loss, norm = self.sam_gradients(inputs, labels)
        # AdamW's moments, clipping and decoupled decay run ONCE at original weights.
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
        self.sam_loss_tracker.update_state(second_loss)
        self.sam_delta_tracker.update_state(second_loss - first[0])
        self.sam_norm_tracker.update_state(norm)
        return self.update(first)


def build_model(cfg, core):
    model = Trainer(cfg, core)
    model({k: tf.zeros((1, *v.shape[1:])) for k, v in core.input.items()})
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"], clipnorm=1.0)
    optimizer.exclude_from_weight_decay(var_names=cfg["wd_exclude"])
    model.compile(optimizer=optimizer, jit_compile=False)
    optimizer.build(model.trainable_variables)
    return model


def predict_views(model, a, b, cfg):
    """Probabilities for the forward view, the RC view and their mean (the reported prediction;
    forward view only when cfg["rc_view"] is False)."""
    probabilities = []
    for view in (a, b):
        out, n = [], len(next(iter(view.values())))
        for start in range(0, n, cfg["prediction_batch_size"]):
            batch = {k: v[start:start + cfg["prediction_batch_size"]] for k, v in view.items()}
            out.append(np.asarray(model(batch, training=False)).ravel())
        probabilities.append(np.concatenate(out))
    if not cfg.get("rc_view", True):
        return *probabilities, probabilities[0]
    return *probabilities, 0.5 * (probabilities[0] + probabilities[1])


def make_dataset(a, b, y, cfg, seed):
    """Seed-dependent shuffle each epoch; batch_size/2 records = batch_size strand views."""
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = (tf.data.Dataset.from_tensor_slices(((a, b), y[:, None].astype(np.float32))).with_options(options)
          .shuffle(len(y), seed=seed, reshuffle_each_iteration=True))
    return ds.batch(cfg["batch_size"] // 2).prefetch(1)


class Selection(tf.keras.callbacks.Callback):
    """EMA-tune-NLL checkpoint rule + LR warmup/halving + early stop; logs the eval-mode train/tune gap."""

    def __init__(self, train, tune, weights_path, cfg):
        super().__init__()
        self.a, self.b, self.y = train
        self.va, self.vb, self.vy = tune
        self.path, self.cfg = weights_path, cfg
        self.best, self.best_epoch, self.ema = np.inf, None, None
        self.wait = self.lr_wait = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.cfg["warmup_epochs"]:
            self.model.optimizer.learning_rate.assign(self.cfg["learning_rate"] * (epoch + 1) / self.cfg["warmup_epochs"])
        self.lr = float(self.model.optimizer.learning_rate.numpy())

    def on_epoch_end(self, epoch, logs=None):
        cfg, ep = self.cfg, epoch + 1
        logs = logs if logs is not None else {}
        pa, pb, p = predict_views(self.model, self.va, self.vb, cfg)
        val = scores(self.vy, p)
        train_acc = train_nll = np.nan
        if ep % cfg["train_eval_every"] == 0 or ep == 1:
            _, _, pt = predict_views(self.model, self.a, self.b, cfg)
            tr = scores(self.y, pt)
            train_acc, train_nll = tr["accuracy"], tr["nll"]
        self.ema = val["nll"] if self.ema is None else cfg["ema_decay"] * self.ema + (1 - cfg["ema_decay"]) * val["nll"]
        logs.update(val_bce=val["nll"], val_bce_ema=self.ema, val_accuracy=val["accuracy"], val_auc=val["roc_auc"],
                    val_rc_gap=float(np.mean(np.abs(pa - pb))), train_eval_accuracy=train_acc,
                    train_eval_nll=train_nll, learning_rate_used=self.lr)
        if not np.isfinite(self.ema) or not np.isfinite(logs.get("loss", np.nan)):
            raise RuntimeError("Non-finite training/tuning values.")
        improved = ep >= cfg["select_start_epoch"] and self.ema < self.best - cfg["min_delta"]
        if improved:
            self.best, self.best_epoch = self.ema, ep
            self.wait = self.lr_wait = 0
            self.model.save_weights(self.path)
        if ep < cfg["select_start_epoch"] or improved:
            return
        self.wait += 1
        self.lr_wait += 1
        if self.lr_wait >= cfg["lr_patience"]:
            self.model.optimizer.learning_rate.assign(max(self.lr * 0.5, cfg["min_lr"]))
            self.lr_wait = 0
        if self.wait >= cfg["patience"]:
            self.model.stop_training = True


def replay_schedule(model, schedule):
    """Refit callback: set the learning rate of every epoch to the base run's logged value."""
    return tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda e, logs: model.optimizer.learning_rate.assign(float(schedule[e])))
