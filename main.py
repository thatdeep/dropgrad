import os
import yaml
import torch
import argparse
import importlib
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter


def save_yaml(dictionary, file_path):
    with open(file_path, 'w') as f:
        yaml.dump(dictionary, f, default_flow_style=False)


def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def namespace_to_dict(ns):
    d = {}
    for key, value in ns.__dict__.items():
        if isinstance(value, SimpleNamespace):
            d[key] = namespace_to_dict(value)
        else:
            d[key] = value
    return d


def import_by_ref(ref):
    parts = ref.split('.')
    module_name, ref_name = '.'.join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_name)
    imported_ref = getattr(module, ref_name)
    return imported_ref


def get_model_instance(config):
    # Initialize Model
    model = import_by_ref(config.model.class_name)().to(config.device)
    return model


def get_dataloaders(config):
    if hasattr(config.data, 'loaders'):
        # import function that will return loaders
        loaders = import_by_ref(config.data.loaders)()
        return loaders
    else:
        raise NotImplementedError('only loaders functions are supported right now. Datasets are wip.')


def get_training_loop(config):
    if hasattr(config.training, 'trainer'):
        training_loop = import_by_ref(config.training.trainer)
        return training_loop
    else:
        raise ValueError('config.training.trainer required to point at some training loop')


def get_test(config):
    if hasattr(config.testing, 'trainer'):
        test = import_by_ref(config.testing.trainer)
        return test
    else:
        raise ValueError('config.testing.trainer rqeuire to point at some testing func')


def get_optimizer_instance(config, model):
    if hasattr(config.training, 'optim'):
        optim_class = import_by_ref(config.training.optim.class_name)
    else:
        import torch.optim.SGD as SGD
        optim_class = SGD
    params_kwargs = {}
    if hasattr(config.training, 'optim') and hasattr(config.training.optim, 'params'):
        params_kwargs = namespace_to_dict(config.training.optim.params)
        #print(f"{config.training.optim.params=}")
        #print(f"{params_kwargs=}")
    optim = optim_class(model.parameters(), **params_kwargs)
    return optim


def get_lr_scheduler_instance(config, optim):
    if hasattr(config.training, 'lr_scheduler'):
        lr_scheduler_class = import_by_ref(config.training.lr_scheduler.class_name)
    else:
        raise ValueError('please specify training.lr_scheduler and its params')
    params_kwargs = {}
    if hasattr(config.training, 'lr_scheduler') and hasattr(config.training.lr_scheduler, 'params'):
        params_kwargs = namespace_to_dict(config.training.lr_scheduler.params)
    lr_scheduler = lr_scheduler_class(optim, **params_kwargs)
    return lr_scheduler


if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--config", type=str, default="experiments/config_mnist.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args.config_path = args.config
        args.config = dict_to_namespace(config)
        logdir = f"runs/{args.config.experiment.log_directory}"
        if not os.path.exists(os.path.join(logdir, "checkpoints")):
            os.makedirs(os.path.join(logdir, "checkpoints"))
        if not os.path.exists(os.path.join(logdir, "tensorboard")):
            os.makedirs(os.path.join(logdir, "tensorboard"))
        save_yaml(config, os.path.join(logdir, "experiment_config.yaml"))
    
    use_cuda = not args.config.gpu.no_cuda and torch.cuda.is_available()
    use_mps = not args.config.gpu.no_mps and torch.backends.mps.is_available()
    print(f"{use_cuda=}, {use_mps=}")
    torch.manual_seed(args.config.random.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args.config.device = device

    # Initialize TensorBoard Writer
    writer = SummaryWriter(os.path.join(logdir, "tensorboard"))

    args.config.writer = writer

    # Load Dataset
    loaders = get_dataloaders(args.config)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Initialize model
    model = get_model_instance(args.config)

    # Initialize optimizer
    optimizer = get_optimizer_instance(args.config, model)

    # Initialize LR scheduler
    scheduler = get_lr_scheduler_instance(args.config, optimizer)

    # Initialize training loop
    training_loop, test = get_training_loop(args.config), get_test(args.config)
    # Training Loop
    # ...
    print(optimizer)
    for epoch in range(1, args.config.training.epochs + 1):
        training_loop(args.config, model, train_loader, optimizer, epoch)
        test(args.config, model, test_loader, epoch)
        if args.config.logging.save_model:
            torch.save(model.state_dict(), os.path.join(logdir, "checkpoints", f"checkpoint_{epoch}.pt"))
        scheduler.step()
