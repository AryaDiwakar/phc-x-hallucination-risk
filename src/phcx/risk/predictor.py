"""Risk predictor models (spec section 13).

Baselines: LogisticRegression, RandomForest, XGBoost, LightGBM, MLP.
Research model: calibrated neural risk predictor (encoder -> fusion -> risk head).
Every model exposes the same interface so ablations/cross-model runs are
trivially composable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from ..utils.io import ensure_dir, save_json

MODEL_NAMES = ("lr", "rf", "xgb", "lgbm", "mlp", "neural")


def _try_xgb():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception:
        return None


def _try_lgbm():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier
    except Exception:
        return None


class RiskModel:
    """Uniform interface over sklearn/xgboost/lightgbm/torch predictors."""

    def __init__(self, name: str = "rf", seed: int = 42, **hyper: Any):
        self.name = name
        self.seed = seed
        self.hyper = hyper
        self.feature_names: List[str] = []
        self.calibrator = None
        self._model = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        h = dict(self.hyper)
        if self.name == "lr":
            self._model = LogisticRegression(
                max_iter=5000, C=h.get("C", 1.0), class_weight="balanced", random_state=self.seed
            )
        elif self.name == "rf":
            self._model = RandomForestClassifier(
                n_estimators=int(h.get("n_estimators", 300)),
                max_depth=h.get("max_depth", 8),
                min_samples_leaf=int(h.get("min_samples_leaf", 5)),
                class_weight="balanced_subsample",
                random_state=self.seed,
                n_jobs=-1,
            )
        elif self.name == "xgb":
            cls = _try_xgb()
            if cls is None:
                raise RuntimeError("xgboost not available")
            self._model = cls(
                n_estimators=int(h.get("n_estimators", 300)),
                max_depth=int(h.get("max_depth", 6)),
                learning_rate=float(h.get("learning_rate", 0.05)),
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.seed,
                n_jobs=-1,
                eval_metric="logloss",
            )
        elif self.name == "lgbm":
            cls = _try_lgbm()
            if cls is None:
                raise RuntimeError("lightgbm not available")
            self._model = cls(
                n_estimators=int(h.get("n_estimators", 300)),
                max_depth=int(h.get("max_depth", 8)),
                learning_rate=float(h.get("learning_rate", 0.05)),
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.name == "mlp":
            self._model = MLPClassifier(
                hidden_layer_sizes=tuple(h.get("hidden_layer_sizes", (64, 32))),
                activation="relu",
                alpha=float(h.get("alpha", 1e-3)),
                learning_rate_init=float(h.get("lr", 1e-3)),
                max_iter=int(h.get("max_iter", 400)),
                early_stopping=True,
                random_state=self.seed,
            )
        elif self.name == "neural":
            self._model = _NeuralRiskPredictor(seed=self.seed, **h)
        else:
            raise ValueError(f"unknown model '{self.name}', expected one of {MODEL_NAMES}")

    # ------------------------------------------------------------------ train
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RiskModel":
        self.feature_names = list(feature_names or [f"f{i}" for i in range(X.shape[1])])
        if self.name == "neural":
            self._model.fit(X, y, X_val=X_val, y_val=y_val, feature_names=self.feature_names)
        else:
            self._model.fit(X, y)
        self._imp_cache = None
        return self

    # ---------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        if self.name == "neural":
            p = self._model.predict_proba(X)
        else:
            p = self._model.predict_proba(X)
            if p.ndim == 2 and p.shape[1] == 1:
                p = np.hstack([1 - p, p])
            p = p[:, 1]
        p = np.asarray(p, dtype=float).reshape(-1)
        if calibrated and self.calibrator is not None:
            p = self.calibrator.transform(p)
        return np.clip(p, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5, calibrated: bool = True) -> np.ndarray:
        return (self.predict_proba(X, calibrated=calibrated) >= threshold).astype(int)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "name": self.name,
            "seed": self.seed,
            "hyper": self.hyper,
            "feature_names": self.feature_names,
        }
        if self.name == "neural":
            self._model.save(path.with_suffix(".pt"))
        else:
            joblib.dump(self._model, path.with_suffix(".joblib"))
        if self.calibrator is not None:
            joblib.dump(self.calibrator, path.parent / f"{path.stem}_calibrator.joblib")
        save_json(path.with_suffix(".json"), payload)

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        obj = cls(name=meta["name"], seed=meta.get("seed", 42), **(meta.get("hyper") or {}))
        obj.feature_names = meta.get("feature_names", [])
        if obj.name == "neural":
            obj._model = _NeuralRiskPredictor(seed=obj.seed, **obj.hyper)
            obj._model.load(path.with_suffix(".pt"))
        else:
            obj._model = joblib.load(path.with_suffix(".joblib"))
        cal_path = path.parent / f"{path.stem}_calibrator.joblib"
        if cal_path.exists():
            obj.calibrator = joblib.load(cal_path)
        return obj

    # ------------------------------------------------------------- explain
    def feature_importance(self) -> Dict[str, float]:
        cached = getattr(self, "_imp_cache", None)
        if cached is not None:
            return cached
        out = self._feature_importance_uncached()
        self._imp_cache = out
        return out

    def _feature_importance_uncached(self) -> Dict[str, float]:
        names = self.feature_names
        if self.name in ("lr",) and hasattr(self._model, "coef_"):
            coefs = np.abs(self._model.coef_).reshape(-1)
            total = float(coefs.sum()) or 1.0
            return {n: float(c / total) for n, c in zip(names, coefs)}
        if self.name == "neural":
            return self._model.feature_importance(names)
        if hasattr(self._model, "feature_importances_"):
            imp = np.asarray(self._model.feature_importances_, dtype=float)
            total = float(imp.sum()) or 1.0
            return {n: float(v / total) for n, v in zip(names, imp)}
        return {n: 0.0 for n in names}


class _NeuralRiskPredictor:
    """Calibrated neural risk predictor: MLP encoder + risk head (sigmoid).

    Trained with BCE + optional temperature-calibrated output on a val split.
    """

    def __init__(
        self,
        seed: int = 42,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 120,
        batch_size: int = 64,
        class_weight_pos: float = 1.0,
        **_: Any,
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.params = dict(
            seed=seed, hidden=hidden, layers=layers, dropout=dropout, lr=lr,
            epochs=epochs, batch_size=batch_size, class_weight_pos=class_weight_pos,
        )
        self.device = torch.device("cpu")

        class Net(nn.Module):
            def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
                super().__init__()
                mods: List[nn.Module] = []
                d = in_dim
                for _ in range(layers):
                    mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                    d = hidden
                self.encoder = nn.Sequential(*mods)
                self.head = nn.Linear(d, 1)

            def forward(self, x):  # type: ignore[override]
                return self.head(self.encoder(x)).squeeze(-1)

        self._Net = Net
        self.net: Optional[nn.Module] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.temperature = 1.0
        self._importance: Dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "_NeuralRiskPredictor":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.params["seed"])
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xs = (X - self.mean) / self.std
        self.net = self._Net(X.shape[1], self.params["hidden"], self.params["layers"],
                             self.params["dropout"]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.params["class_weight_pos"]], dtype=torch.float32)
        )
        xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        n = len(Xs)
        bs = max(8, int(self.params["batch_size"]))
        for _ in range(int(self.params["epochs"])):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i: i + bs]
                opt.zero_grad()
                logits = self.net(xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()
        # temperature scaling on validation logits (calibration-friendly)
        if X_val is not None and y_val is not None and len(X_val):
            with torch.no_grad():
                xv = torch.tensor((X_val - self.mean) / self.std, dtype=torch.float32)
                logits = self.net(xv).numpy()
            self.temperature = _fit_temperature(logits, y_val)
        if feature_names is not None:
            self._gradient_importance(Xs, feature_names)
        return self

    def _gradient_importance(self, Xs: np.ndarray, names: Sequence[str]) -> None:
        try:
            import torch

            assert self.net is not None and len(Xs) > 0
            xt = torch.tensor(Xs[: min(256, len(Xs))], dtype=torch.float32, requires_grad=True)
            out = torch.sigmoid(self.net(xt) / self.temperature).mean()
            out.backward()
            grad = (xt.grad.abs() * xt).abs().mean(dim=0).numpy()
            total = float(grad.sum()) or 1.0
            self._importance = {n: float(v / total) for n, v in zip(names, grad)}
        except Exception:
            self._importance = {n: 0.0 for n in names}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        assert self.net is not None and self.mean is not None
        self.net.eval()
        with torch.no_grad():
            xs = torch.tensor((X - self.mean) / self.std, dtype=torch.float32)
            logits = self.net(xs).numpy() / max(1e-6, self.temperature)
        return 1.0 / (1.0 + np.exp(-logits))

    def feature_importance(self, names: Sequence[str]) -> Dict[str, float]:
        if self._importance:
            return dict(self._importance)
        return {n: 0.0 for n in names}

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        ensure_dir(path.parent)
        assert self.net is not None
        torch.save(
            {"state_dict": self.net.state_dict(), "params": self.params, "mean": self.mean,
             "std": self.std, "temperature": self.temperature, "in_dim": len(self.mean)},
            path,
        )

    def load(self, path: str | Path) -> "_NeuralRiskPredictor":
        import torch

        blob = torch.load(path, map_location="cpu")
        self.params = blob["params"]
        self.mean, self.std = blob["mean"], blob["std"]
        self.temperature = float(blob.get("temperature", 1.0))
        self.net = self._Net(int(blob["in_dim"]), self.params["hidden"], self.params["layers"],
                             self.params["dropout"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        return self


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit a single scalar temperature by minimizing NLL (grid search)."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.arange(0.25, 3.01, 0.05)]):
        z = logits / t
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t
