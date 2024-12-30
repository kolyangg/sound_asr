def poly_decay_power_05(step, total_steps):
    # if step > total_steps, clamp at 0
    ratio = max(0.0, 1.0 - float(step)/float(total_steps))
    return ratio**0.5


def polynomial_decay_epoch(epoch, max_epochs, power=0.5):
    """
    Polynomial decay factor for epoch-based schedule.
    epoch: Current epoch (0-based)
    max_epochs: Total epochs for training
    power: Exponent for polynomial decay (default=0.5)
    Returns a decay factor in [0,1].
    """
    # ratio goes from 1.0 at epoch=0 down to 0.0 at epoch=max_epochs
    ratio = 1.0 - float(epoch) / float(max_epochs)
    if ratio < 0:
        ratio = 0.0
    return ratio ** power
