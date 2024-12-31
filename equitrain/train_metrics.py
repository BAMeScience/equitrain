class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def log_metrics(args, logger, prefix, postfix, loss_metrics):

    info_str  = prefix
    info_str += 'loss: {loss:.5f}'.format(loss=loss_metrics['total'].avg)

    if args.energy_weight > 0.0:
        info_str += ', loss_e: {loss_e:.5f}'.format(
            loss_e=loss_metrics['energy'].avg,
        )
    if args.forces_weight > 0.0:
        info_str += ', loss_f: {loss_f:.5f}'.format(
            loss_f=loss_metrics['forces'].avg,
        )
    if args.stress_weight > 0.0:
        info_str += ', loss_s: {loss_f:.5f}'.format(
            loss_f=loss_metrics['stress'].avg,
        )

    if postfix is not None:
        info_str += postfix

    logger.log(1, info_str)


def update_best_results(criterion, best_metrics, val_loss, epoch):

    update_result = False

    loss_new = val_loss['total'].avg
    loss_old = best_metrics['loss']

    if loss_new < loss_old:

        best_metrics['loss'] = val_loss['total'].avg

        if criterion.energy_weight > 0.0:
            best_metrics['energy_loss'] = val_loss['energy'].avg
        if criterion.forces_weight > 0.0:
            best_metrics['forces_loss'] = val_loss['forces'].avg
        if criterion.stress_weight > 0.0:
            best_metrics['stress_loss'] = val_loss['stress'].avg

        best_metrics['epoch'] = epoch

        update_result = True

    return update_result
