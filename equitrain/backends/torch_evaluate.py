import csv
import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from equitrain.argparser import ArgsFormatter, get_loss_monitor, validate_evaluate_args
from equitrain.data.backend_torch.loaders import get_dataloader
from equitrain.logger import init_logger

from .torch_loss import LossCollection
from .torch_loss_fn import LossFnCollection
from .torch_loss_metrics import LossMetrics
from .torch_model import get_model
from .torch_utils import set_dtype

warnings.filterwarnings('ignore', message=r'.*TorchScript type system.*')


def _metric_to_dict(metric) -> dict[str, dict[str, float]]:
    result = {}
    for name, meter in metric.items():
        if meter is None:
            continue
        result[name] = {
            'avg': float(meter.avg),
            'sum': float(meter.sum),
            'count': float(meter.count),
        }
    return result


def _loss_metrics_to_dict(loss_metrics: LossMetrics) -> dict[str, dict]:
    result = {loss_metrics.main_type: _metric_to_dict(loss_metrics.main)}
    for loss_type, metric in loss_metrics.items():
        result[loss_type] = _metric_to_dict(metric)
    return result


def _write_evaluation_results(
    args,
    loss_metrics: LossMetrics,
    errors: torch.Tensor,
    logger,
) -> None:
    output_dir = getattr(args, 'output_dir', None)
    if not output_dir:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    errors_path = output_path / 'test_errors.csv'
    with errors_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['index', 'error'])
        for index, error in enumerate(errors.detach().cpu().tolist()):
            writer.writerow([index, float(error)])

    metrics_path = output_path / 'test_metrics.json'
    payload = {
        'backend': 'torch',
        'dataset': args.test_file,
        'loss_type': loss_metrics.main_type,
        'metrics': _loss_metrics_to_dict(loss_metrics),
        'errors_file': errors_path.name,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.log(1, f'Wrote evaluation metrics to `{metrics_path}`')


def evaluate_main(
    args,
    model: torch.nn.Module,
    accelerator: Accelerator,
    dataloader: Iterable,
    max_steps: int = None,
    desc: str = 'Evaluating',
):
    loss_fn = LossFnCollection(**vars(args))

    model.eval()

    loss_metrics = LossMetrics(args)

    errors = torch.zeros(len(dataloader.dataset), device=accelerator.device)

    if max_steps is None:
        total = len(dataloader)
    else:
        total = max_steps

    with tqdm(
        dataloader,
        total=total,
        disable=not args.tqdm or not accelerator.is_main_process,
        desc=desc,
    ) as pbar:
        for step, data_list in enumerate(pbar):
            # Compute a collection of loss metrics for monitoring purposes
            loss_collection = LossCollection(
                args.loss_monitor, device=accelerator.device
            )

            with accelerator.no_sync(model):
                for data in data_list:
                    y_pred = model(data)

                    loss, error = loss_fn(y_pred, data)

                    if loss.main.isfinite():
                        loss_collection += loss
                        error = error.to(dtype=errors.dtype, device=errors.device)
                        errors[data.idx] = error

            # Gather loss across processes for computing metrics
            loss_for_metrics = loss_collection.gather_for_metrics(accelerator)

            loss_metrics.update(loss_for_metrics)

            if accelerator.is_main_process and args.tqdm:
                pbar.set_description(
                    f'{desc} (loss={loss_metrics.main["total"].avg:.04g})'
                )

            # Stop evaluating if maximum number of steps is defined and reached
            if step >= total:
                break

    # Sum local errors across all processes
    accelerator.reduce(errors, reduction='sum')

    return loss_metrics, errors


def _evaluate_with_accelerator(args, accelerator: Accelerator):
    # Only main process should output information
    logger = init_logger(
        args,
        backend_name='torch',
        enable_logging=accelerator.is_main_process,
        log_to_file=False,
        output_dir=None,
    )
    logger.log(1, ArgsFormatter(args))

    """ Network """
    model = get_model(args, logger=logger)

    """ Data Loader """
    data_loader = get_dataloader(
        args, args.test_file, model.atomic_numbers, model.r_max, accelerator
    )

    # Prepare all components with accelerate
    model = accelerator.prepare(model)

    test_loss, errors = evaluate_main(
        args,
        model=model,
        accelerator=accelerator,
        dataloader=data_loader,
    )

    if accelerator.is_main_process:
        test_loss.log(logger, 'test', epoch=None, force=True)
        _write_evaluation_results(args, test_loss, errors, logger)

    return test_loss


def evaluate_(args):
    set_dtype(args.dtype)

    if args.energy_weight == 0.0:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    # Filter out main loss type and convert to list
    args.loss_monitor = get_loss_monitor(args)

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    try:
        return _evaluate_with_accelerator(args, accelerator)

    finally:
        accelerator.end_training()


def evaluate(args):
    validate_evaluate_args(args, 'torch')
    return evaluate_(args)
