from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from unittest import TestCase
import json
import torch
import pytorch_lightning as pl

from openmapflow.config import PROJECT_ROOT, DataPaths

from src.models.model import Model


class IntegrationTestModelEvaluation(TestCase):

    scores: List[Tuple[Any, ...]] = []

    @classmethod
    def setUpClass(cls) -> None:

        model_dir = PROJECT_ROOT / DataPaths.MODELS
        with (PROJECT_ROOT / DataPaths.METRICS).open("rb") as f:
            models_dict: Dict[str, Any] = json.load(f)

        for model_name, model_dict in tqdm(models_dict.items()):

            recorded_f1 = model_dict["val_metrics"]["f1_score"]

            if not (model_dir / f"{model_name}.ckpt").exists():
                cls.scores.append((model_name, recorded_f1, None, None, None))
                continue

            model_ckpt = Model.load_from_checkpoint(model_dir / f"{model_name}.ckpt")
            model_ckpt.eval()

            # Get validation set
            val = model_ckpt.get_dataset(
                subset="validation",
                normalizing_dict=model_ckpt.normalizing_dict,
                upsample=False,
                cache=False,
            )

            # Get tensors from validation set
            x = torch.stack([v[0] for v in val])
            y_true = torch.stack([v[1] for v in val])

            # Feed tensors into both models
            with torch.no_grad():
                y_pred_ckpt = model_ckpt(x).numpy()

            y_pred_ckpt_binary = y_pred_ckpt > 0.5

            ckpt_f1 = round(f1_score(y_true=y_true, y_pred=y_pred_ckpt_binary), 4)

            trainer = pl.Trainer(checkpoint_callback=False, logger=False)
            trainer.model = model_ckpt
            trainer.main_progress_bar = tqdm(disable=True)
            trainer.run_evaluation(test_mode=False)

            trainer_f1 = round(trainer.callback_metrics["f1_score"], 4)

            cls.scores.append((model_name, recorded_f1, ckpt_f1, trainer_f1, None))

            # if not (model_dir / f"{model_name}.pt").exists():
            #     cls.scores.append((model_name, recorded_f1, ckpt_f1, trainer_f1, None))
            #     continue

            # model_pt = torch.jit.load(str(model_dir / f"{model_name}.pt"))
            # model_pt.eval()

            # with torch.no_grad():
            #     y_pred_pt = model_pt(x)[1].numpy()

            # y_pred_pt_binary = y_pred_pt > 0.5
            # pt_f1 = round(f1_score(y_true=y_true, y_pred=y_pred_pt_binary), 4)

            # cls.scores.append((model_name, recorded_f1, ckpt_f1, trainer_f1, pt_f1))

    def test_model_eval(self):
        no_differences = True
        for model_name, recorded_f1, ckpt_f1, trainer_f1, pt_f1 in self.scores:
            print("---------------------------------------------")
            print(model_name)
            if recorded_f1 == ckpt_f1:
                print(f"\u2714 Recorded F1 == CKPT F1 == {ckpt_f1}")
            else:
                no_differences = False
                print(f"\u2716 Recorded F1: {recorded_f1} != CKPT F1 {ckpt_f1}")
            if ckpt_f1 == trainer_f1:
                print(f"\u2714 CKPT F1 == trainer F1 == {trainer_f1}")
            else:
                no_differences = False
                print(f"\u2716 CKPT F1: {ckpt_f1} != trainer F1 {trainer_f1}")
            if pt_f1:
                if ckpt_f1 == pt_f1:
                    print(f"\u2714 CKPT F1 == PT F1 == {pt_f1}")
                else:
                    no_differences = False
                    print(f"\u2716 CKPT F1: {ckpt_f1} != PT F1 {pt_f1}")

        self.assertTrue(no_differences, "Some ckpt models don't match, check logs.")
