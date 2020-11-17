"""Utility function."""


def decode_label(y_pred):
    label = y_pred[0].astype(int)
    if label == 0:
        return "not_duplicate"
    else:
        return "duplicate"
