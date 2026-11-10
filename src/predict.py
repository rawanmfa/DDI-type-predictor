"""Loads the trained model and predicts the interaction type for a pair of known drugs."""
import json
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from .model import MMADL


def fill_template(template, drug_a, drug_b):
    """Event templates contain the word 'name' where a drug name goes: the first one is
    drug A (the drug that is affected), the second one is drug B."""
    names = iter([drug_a, drug_b])
    return re.sub(r"\bname\b", lambda m: next(names, drug_b), template)


class Predictor:
    def __init__(self, artifact_dir="models"):
        with open(os.path.join(artifact_dir, "meta.json")) as fh:
            meta = json.load(fh)
        self.meta = meta
        self.ids = meta["drug_ids"]
        self.names = meta["drug_names"]
        self.class_names = meta["class_names"]
        self.idx = {d: i for i, d in enumerate(self.ids)}

        views = {n: torch.zeros(len(self.ids), dim) for n, dim in meta["view_dims"].items()}
        self.model = MMADL(views, meta["num_classes"], **meta["model_config"])
        state = torch.load(os.path.join(artifact_dir, "model.pt"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.no_grad():
            self.z = self.model.encode_drugs()  # one vector per drug, computed once

        pairs = np.load(os.path.join(artifact_dir, "known_pairs.npz"))
        self.known = {(int(a), int(b)): int(c)
                      for a, b, c in zip(pairs["i1"], pairs["i2"], pairs["y"])}

        self.choice_to_id = {f"{n} ({i})": i for n, i in zip(self.names, self.ids)}

    def drug_choices(self):
        return sorted(self.choice_to_id)

    @torch.no_grad()
    def predict(self, id_a, id_b, top_k=3):
        if id_a == id_b:
            raise ValueError("Choose two different drugs.")
        i, j = self.idx[id_a], self.idx[id_b]
        feat = torch.cat([self.z[i], self.z[j], self.z[i] * self.z[j]]).unsqueeze(0)
        probs = F.softmax(self.model.decoder(feat), dim=1)[0]
        top = torch.topk(probs, min(top_k, probs.numel()))
        name_a, name_b = self.names[i], self.names[j]
        predictions = [{"class_id": int(c),
                        "text": fill_template(self.class_names[int(c)], name_a, name_b),
                        "probability": float(p)}
                       for p, c in zip(top.values, top.indices)]
        known = self.known.get((i, j))
        return {
            "drug_a": name_a,
            "drug_b": name_b,
            "predictions": predictions,
            "in_dataset": known is not None,
            "dataset_label": (fill_template(self.class_names[known], name_a, name_b)
                              if known is not None else None),
        }