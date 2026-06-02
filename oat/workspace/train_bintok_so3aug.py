if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import numpy as np
import random
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pathlib
import copy
import tqdm

from oat.dataset.base_dataset import BaseDataset
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.hydra_util import register_new_resolvers
from oat.tokenizer.bin.tokenizer_so3_aug import BinTokSO3Aug
from oat.workspace.train_bintok import TrainBinTokWorkspace

register_new_resolvers()


class TrainBinTokSO3AugWorkspace(TrainBinTokWorkspace):
    """BinTok eval workspace that applies SO(3) augmentation to eval actions.

    The bin tokenizer is parameter-free, so the augmentation is applied to the
    raw validation actions before tokenize/detokenize. The reconstruction MSE is
    then measured against the augmented actions, mirroring the augmented OATTok
    eval.
    """

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # set seed
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model after seeding
        self.model: BinTokSO3Aug = hydra.utils.instantiate(cfg.tokenizer)

        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(
            cfg.task.tokenizer.dataset)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure normalizer
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk
        )

        # reconstruction eval on SO(3)-augmented actions
        self.model.eval()
        step_log = dict()
        with torch.inference_mode():
            loss_info = torch.zeros(2, dtype=torch.float32)
            with tqdm.tqdm(
                val_dataloader,
                desc="Reconstruction eval",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    samples = batch['action']   # (B, T, D)
                    samples = self.model.augment(samples)
                    reconst_samples = self.model.detokenize(self.model.tokenize(samples))
                    mse = F.mse_loss(reconst_samples, samples).item()

                    batch_size = samples.shape[0]
                    loss_info[0] += mse * batch_size
                    loss_info[1] += batch_size

                    if (cfg.training.max_reconst_steps is not None) \
                        and batch_idx >= (cfg.training.max_reconst_steps - 1):
                        break

            step_log['test_reconst_mse'] = (loss_info[0] / loss_info[1]).item()
            print(f"Reconstruction MSE: {step_log['test_reconst_mse']}")

        # checkpoint
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()
        metric_dict = dict()
        for k, v in step_log.items():
            new_key = k.replace('/', '_')
            metric_dict[new_key] = v
        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
        if topk_ckpt_path is not None:
            self.save_checkpoint(path=topk_ckpt_path)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainBinTokSO3AugWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
