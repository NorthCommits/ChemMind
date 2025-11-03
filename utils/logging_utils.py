import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def create_tensorboard_writer(runs_dir: str) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(runs_dir, timestamp)
    os.makedirs(logdir, exist_ok=True)
    return SummaryWriter(log_dir=logdir)


