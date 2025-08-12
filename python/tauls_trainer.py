#!/usr/bin/env python3
"""
Enhanced TA ULS Training System with Julia Integration
Implements stability-aware training with polynomial optimization backend
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim


class JuliaClient:
    """Python client for Julia mathematical operations (backed by our FastAPI mock)."""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()

    def _make_request(self, function_name: str, args: List[Any]) -> Dict[str, Any]:
        try:
            payload = {"function": function_name, "args": args}
            response = self.session.post(self.server_url, json=payload, headers={"Content-Type": "application/json"})
            return response.json() if response.status_code == 200 else {"error": f"Server error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

    def optimize_matrix(self, matrix: np.ndarray, method: str = "sparsity") -> Dict[str, Any]:
        return self._make_request("optimize_matrix", [matrix.tolist(), method])

    def create_polynomials(self, data: np.ndarray, variables: List[str]) -> Dict[str, Any]:
        return self._make_request("create_polynomials", [data.tolist(), variables])

    def analyze_polynomials(self, polynomials: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request("analyze_polynomials", [polynomials])


class KFPLayer(nn.Module):
    def __init__(self, dim: int, stability_weight: float = 0.1):
        super().__init__()
        self.dim = dim
        self.stability_weight = stability_weight
        self.fluctuation_history = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.momentum = 0.9
        self.force_projection = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reduce variance across all non-feature axes so the result matches feature dim
        if x.dim() <= 1:
            current_fluctuation = torch.zeros(self.dim, device=x.device, dtype=x.dtype)
        else:
            reduce_dims = tuple(range(0, x.dim() - 1))  # all except last
            current_fluctuation = torch.var(x, dim=reduce_dims, keepdim=False)
            if current_fluctuation.dim() == 0:
                current_fluctuation = current_fluctuation * torch.ones(self.dim, device=x.device, dtype=x.dtype)
        # Momentum update of fluctuation history
        self.fluctuation_history.data = (
            self.momentum * self.fluctuation_history.data
            + (1 - self.momentum) * current_fluctuation.detach()
        )
        # Linear projection and stability push
        kinetic_force = self.force_projection(x)
        stability_term = -self.stability_weight * kinetic_force
        return x + stability_term, self.fluctuation_history


class TAULSControlUnit(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, control_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        self.meta_controller = nn.Sequential(
            nn.Linear(input_dim + control_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            KFPLayer(hidden_dim),
            nn.Linear(hidden_dim, control_dim),
        )

        self.controller = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            KFPLayer(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, control_dim),
        )

        self.control_mixer = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, prev_control: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        batch_size, seq_len = x.shape[:2]

        if prev_control is None:
            prev_control = torch.zeros(batch_size, seq_len, self.control_dim, device=x.device)

        meta_input = torch.cat([x, prev_control], dim=-1)
        meta_h, meta_stability = self.meta_controller[:-1](meta_input.reshape(-1, meta_input.shape[-1]))
        meta_control = self.meta_controller[-1](meta_h).reshape(batch_size, seq_len, -1)

        auto_h, auto_stability = self.controller[:-1](x.reshape(-1, x.shape[-1]))
        auto_control = self.controller[-1](auto_h).reshape(batch_size, seq_len, -1)

        alpha = torch.sigmoid(self.control_mixer)
        integrated_control = alpha * meta_control + (1 - alpha) * auto_control

        return {
            "control_output": integrated_control,
            "meta_stability": meta_stability,
            "auto_stability": auto_stability,
            "control_mixing": alpha,
        }


@dataclass
class TAULSTrainingConfig:
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 2048
    batch_size: int = 8
    learning_rate: float = 1e-4
    stability_weight: float = 0.1
    entropy_weight: float = 0.05
    julia_server_port: int = 8000
    use_julia_optimization: bool = True
    optimization_frequency: int = 100
    stability_threshold: float = 0.8
    max_entropy_target: float = 0.7


class JuliaServerManager:
    def __init__(self, port: int = 8000):
        self.port = port
        self.client: Optional[JuliaClient] = None

    def start_server(self) -> bool:
        # In our environment, we use the FastAPI mock started separately.
        # Just wire the client to the configured port and check health by a simple call.
        try:
            self.client = JuliaClient(f"http://localhost:{self.port}")
            # quick ping by calling a trivial function
            res = self.client.optimize_matrix(np.random.rand(2, 2))
            ok = "error" not in res
            if ok:
                logging.info(f"Mock Julia server available on port {self.port}")
            else:
                logging.error(f"Mock Julia server check failed: {res}")
            return ok
        except Exception as e:
            logging.error(f"Failed to connect to mock Julia server: {e}")
            return False

    def stop_server(self):
        # Nothing to do for the mock
        pass


class StabilityAwareLoss(nn.Module):
    def __init__(self, entropy_weight: float = 0.05, stability_weight: float = 0.1):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.stability_weight = stability_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_entropy_loss(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        total_entropy = 0.0
        for hidden in hidden_states:
            entropy = torch.var(hidden, dim=-1).mean()
            total_entropy += entropy
        return total_entropy / max(len(hidden_states), 1)

    def compute_stability_loss(self, stability_metrics: List[Dict[str, Any]]) -> torch.Tensor:
        total_stability_loss = 0.0
        count = 0
        for metrics in stability_metrics:
            if "stability_info" in metrics:
                stability_info = metrics["stability_info"]
                total_stability_loss += torch.mean(stability_info)
                count += 1
        return total_stability_loss / max(count, 1)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: List[torch.Tensor],
        stability_metrics: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        ce = self.ce_loss(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        ent = self.compute_entropy_loss(hidden_states)
        stab = self.compute_stability_loss(stability_metrics)
        total = ce + self.entropy_weight * ent + self.stability_weight * stab
        return {"total_loss": total, "ce_loss": ce, "entropy_loss": ent, "stability_loss": stab}


class TAULSOptimizer:
    def __init__(self, model: nn.Module, config: TAULSTrainingConfig, julia_client: Optional[JuliaClient]):
        self.model = model
        self.config = config
        self.julia_client = julia_client
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.step_count = 0

    def optimize_parameters_with_julia(self) -> Dict[str, Any]:
        if self.julia_client is None:
            return {}
        optimized_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_np = param.data.detach().cpu().numpy()
                if param_np.size < 4 or param_np.ndim < 2:
                    continue
                flat = param_np.reshape(param_np.shape[0], -1)
                result = self.julia_client.optimize_matrix(flat, method="sparsity")
                if "error" not in result and "optimized_matrix" in result:
                    opt = np.array(result["optimized_matrix"], dtype=param_np.dtype)
                    opt = opt.reshape(flat.shape)
                    if param_np.ndim > 2:
                        opt = opt.reshape(param_np.shape)
                    alpha = 0.1
                    mixed = (1 - alpha) * param_np + alpha * opt
                    param.data = torch.as_tensor(mixed, dtype=param.dtype, device=param.device)
                    optimized_params[name] = {
                        "compression_ratio": result.get("compression_ratio", 0.0),
                        "optimization_method": result.get("method", "sparsity"),
                    }
        return optimized_params

    def step(self, loss: torch.Tensor) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimization_info: Dict[str, Any] = {}
        if self.julia_client is not None and (self.step_count % self.config.optimization_frequency == 0):
            optimization_info = self.optimize_parameters_with_julia()
        self.optimizer.step()
        self.step_count += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float("inf")))
        return {"step": self.step_count, "julia_optimization": optimization_info, "gradient_norm": grad_norm}


class TAULSTrainer:
    def __init__(self, config: TAULSTrainingConfig):
        self.config = config
        self.julia_manager = JuliaServerManager(port=config.julia_server_port)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model().to(self.device)
        self.loss_fn = StabilityAwareLoss(config.entropy_weight, config.stability_weight)
        self.optimizer: Optional[TAULSOptimizer] = None

    def _create_model(self) -> nn.Module:
        class SimpleTA_ULS(nn.Module):
            def __init__(self, config: TAULSTrainingConfig):
                super().__init__()
                self.embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.control_unit = TAULSControlUnit(config.d_model, config.d_model, config.d_model)
                self.output = nn.Linear(config.d_model, config.vocab_size)
                self.kfp_layer = KFPLayer(config.d_model)

            def forward(self, input_ids: torch.Tensor) -> Dict[str, Any]:
                x = self.embedding(input_ids)
                control_result = self.control_unit(x)
                controlled_x = control_result["control_output"]
                stable_x, stability_info = self.kfp_layer(controlled_x)
                logits = self.output(stable_x)
                return {
                    "logits": logits,
                    "hidden_states": [x, controlled_x, stable_x],
                    "stability_metrics": [{"stability_info": stability_info}],
                    "control_info": control_result,
                }

        return SimpleTA_ULS(self.config)

    def start_training(self) -> None:
        if self.config.use_julia_optimization:
            if not self.julia_manager.start_server():
                logging.warning("Julia mock server not available, disabling Julia optimization")
                self.config.use_julia_optimization = False
        client = self.julia_manager.client if self.config.use_julia_optimization else None
        self.optimizer = TAULSOptimizer(self.model, self.config, client)
        logging.info("Training environment initialized")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        assert self.optimizer is not None
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        out = self.model(input_ids)
        loss_info = self.loss_fn(out["logits"], targets, out["hidden_states"], out["stability_metrics"])
        opt_info = self.optimizer.step(loss_info["total_loss"]) 
        return {"loss": loss_info, "optimization": opt_info, "model_output": {k: v for k, v in out.items() if k != "logits"}}

    def evaluate_stability(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            test_input = torch.randint(0, self.config.vocab_size, (self.config.batch_size, 64), device=self.device)
            outputs = [self.model(test_input) for _ in range(5)]
            logit_variance = torch.var(torch.stack([o["logits"] for o in outputs]), dim=0).mean()
            scores: List[float] = []
            for output in outputs:
                for metric in output["stability_metrics"]:
                    if "stability_info" in metric:
                        scores.append(metric["stability_info"].mean().item())
            return {
                "logit_stability": float(logit_variance.item()),
                "mean_stability_score": float(np.mean(scores)) if scores else 0.0,
                "stability_variance": float(np.var(scores)) if scores else 0.0,
            }

    def cleanup(self) -> None:
        self.julia_manager.stop_server()


def create_dummy_dataset(config: TAULSTrainingConfig, num_samples: int = 100) -> List[Dict[str, torch.Tensor]]:
    data: List[Dict[str, torch.Tensor]] = []
    for _ in range(num_samples):
        seq_len = int(np.random.randint(10, min(config.max_seq_len, 100)))
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        targets = torch.randint(0, config.vocab_size, (1, seq_len))
        data.append({"input_ids": input_ids, "targets": targets})
    return data


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = TAULSTrainingConfig(vocab_size=1000, d_model=128, batch_size=4, use_julia_optimization=True, optimization_frequency=5)
    trainer = TAULSTrainer(config)
    try:
        trainer.start_training()
        dataset = create_dummy_dataset(config, num_samples=20)
        for step, batch in enumerate(dataset):
            result = trainer.train_step(batch)
            if step % 5 == 0:
                logging.info(f"step {step} total_loss={result['loss']['total_loss'].item():.4f} grad_norm={result['optimization']['gradient_norm']:.3f}")
        stability = trainer.evaluate_stability()
        logging.info(f"stability: {stability}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()