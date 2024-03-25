import math

def sigmoid_rampup(epoch, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = min(epoch, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(math.exp(-5.0 * phase * phase))
