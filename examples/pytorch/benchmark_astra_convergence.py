# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Controlled ASTRA solver benchmark on a tiny exact-GGN problem.

This is a library convergence benchmark, not a reproduction of the paper's LDS
experiments. It compares all methods against an explicitly materialized damped GGN.
"""
import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deel.influenciae.common import AstraConfig, AstraIHVP, EkfacIHVP, InfluenceModel
from deel.influenciae.common.ggn import GeneralizedGaussNewtonOperator


def _parse_iterations(value: str) -> List[int]:
    iterations = sorted({int(item) for item in value.split(",")})
    if not iterations or iterations[0] < 0:
        raise argparse.ArgumentTypeError("iterations must be comma-separated non-negative integers")
    return iterations


def _relative_norm(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return float((torch.linalg.vector_norm(value) / denominator).item())


def _measure(
    name: str,
    solution: torch.Tensor,
    exact_solution: torch.Tensor,
    system: torch.Tensor,
    rhs: torch.Tensor,
    seconds: float,
    iterations: int,
    factor_seconds: float,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    residual = solution @ system.T - rhs
    row: Dict[str, object] = {
        "method": name,
        "iterations": iterations,
        "curvature_batches": iterations,
        "relative_solution_error": _relative_norm(solution - exact_solution, exact_solution),
        "relative_residual": _relative_norm(residual, rhs),
        "solve_seconds": seconds,
        "factor_build_seconds": factor_seconds,
    }
    row.update(metadata)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=5)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=_parse_iterations, default=_parse_iterations("0,1,10,50"))
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--preconditioner-damping", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--sni-learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if min(args.samples, args.input_dim, args.hidden_dim, args.output_dim, args.batch_size) <= 0:
        parser.error("model, sample, and batch dimensions must be positive")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)

    model = nn.Sequential(
        nn.Linear(args.input_dim, args.hidden_dim, dtype=dtype),
        nn.Tanh(),
        nn.Linear(args.hidden_dim, args.output_dim, dtype=dtype),
    )
    inputs = torch.randn(args.samples, args.input_dim, dtype=dtype)
    targets = torch.randn(args.samples, args.output_dim, dtype=dtype)
    dataset = DataLoader(TensorDataset(inputs, targets), batch_size=args.batch_size, shuffle=False)
    influence_model = InfluenceModel(
        model,
        start_layer=0,
        last_layer=-1,
        loss_function=nn.MSELoss(reduction="none"),
    )
    full_batch = (inputs, targets)
    parameter_count = influence_model.nb_params
    ggn = GeneralizedGaussNewtonOperator(influence_model, full_batch, "mean")
    basis = torch.eye(parameter_count, dtype=dtype)
    dense_ggn = ggn.matmat(basis)
    system = dense_ggn + args.damping * basis
    rhs = torch.randn(1, parameter_count, dtype=dtype)
    exact_solution = torch.linalg.solve(system, rhs.T).T

    factor_start = perf_counter()
    ekfac = EkfacIHVP(
        influence_model,
        dataset,
        damping=args.preconditioner_damping or args.damping,
    )
    factor_seconds = perf_counter() - factor_start

    solve_start = perf_counter()
    ekfac_solution = ekfac.precondition_gradient(rhs).T
    ekfac_seconds = perf_counter() - solve_start
    metadata = {
        "seed": args.seed,
        "dtype": args.dtype,
        "damping": args.damping,
        "preconditioner_damping": args.preconditioner_damping or args.damping,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "samples": args.samples,
        "parameters": parameter_count,
    }
    rows = [
        _measure(
            "exact", exact_solution, exact_solution, system, rhs, 0.0, 0,
            0.0, metadata,
        ),
        _measure(
            "ekfac", ekfac_solution, exact_solution, system, rhs, ekfac_seconds, 0,
            factor_seconds, metadata,
        ),
    ]

    for iterations in args.iterations:
        config = AstraConfig(
            damping=args.damping,
            preconditioner_damping=args.preconditioner_damping,
            n_iterations=iterations,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            initialize_from_ekfac=True,
            seed=args.seed,
        )
        astra = AstraIHVP(
            influence_model,
            dataset,
            config=config,
            ekfac_factors=ekfac.factors,
            curvature_batch_sampler=lambda _step: full_batch,
        )
        solve_start = perf_counter()
        astra_solution = astra.precondition_gradient(rhs).T
        astra_seconds = perf_counter() - solve_start
        rows.append(_measure(
            f"astra-{iterations}", astra_solution, exact_solution, system, rhs,
            astra_seconds, iterations, factor_seconds, metadata,
        ))

        solve_start = perf_counter()
        sni_solution = torch.zeros_like(rhs)
        for _ in range(iterations):
            residual = ggn.matmat(sni_solution) + args.damping * sni_solution - rhs
            sni_solution = sni_solution - args.sni_learning_rate * residual
        sni_seconds = perf_counter() - solve_start
        rows.append(_measure(
            f"sni-{iterations}", sni_solution, exact_solution, system, rhs,
            sni_seconds, iterations, 0.0, metadata,
        ))

    print(f"{'method':<14} {'iters':>7} {'rel_error':>14} {'rel_residual':>14} {'seconds':>11}")
    for row in rows:
        print(
            f"{row['method']:<14} {row['iterations']:>7} "
            f"{row['relative_solution_error']:>14.6e} "
            f"{row['relative_residual']:>14.6e} {row['solve_seconds']:>11.6f}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
