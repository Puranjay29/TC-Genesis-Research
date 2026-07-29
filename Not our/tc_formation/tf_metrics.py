"""
A workaround for working with metrics that don't support `from_logits=True` output.
"""
import tensorflow as tf

class FromLogitsMixin:
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)

class PrecisionScore(FromLogitsMixin, tf.metrics.Precision):
    pass

class RecallScore(FromLogitsMixin, tf.metrics.Recall):
    pass

class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', class_id=None, thresholds=0.5, from_logits=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision_fn = PrecisionScore(thresholds=thresholds, class_id=class_id, from_logits=from_logits)
        self.recall_fn = RecallScore(thresholds=thresholds, class_id=class_id, from_logits=from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_fn.update_state(y_true, y_pred)
        self.recall_fn.update_state(y_true, y_pred)

    def result(self):
        p = self.precision_fn.result()
        r = self.recall_fn.result()
        return 2 * p * r / (p + r + 1e-6)

    def reset_state(self):
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()