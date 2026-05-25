from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time
import re

import numpy as np


@dataclass
class DHFSPInstance:
    name: str
    jobs: int
    stages: int
    factories: int
    machines_per_stage: list[int]


    processing_times: np.ndarray
    due_dates: np.ndarray
    workloads: np.ndarray | None = None
    machine_efficiencies: np.ndarray | None = None
    source_path: Path | None = None


@dataclass
class Solution:





    permutation: np.ndarray
    factory_assignment: np.ndarray
    objectives: tuple[float, float] | None = None
    rank: int = 0
    crowding_distance: float = 0.0
    completion_times: np.ndarray | None = None
    machine_on_time: float | None = None
    processing_time_total: float | None = None

    def clone(self) -> "Solution":
        return Solution(
            permutation=self.permutation.copy(),
            factory_assignment=self.factory_assignment.copy(),
            objectives=self.objectives,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            completion_times=None if self.completion_times is None else self.completion_times.copy(),
            machine_on_time=self.machine_on_time,
            processing_time_total=self.processing_time_total,
        )


@dataclass(frozen=True)
class RunConfig:
    instance_path: Path
    output_dir: Path
    population_size: int
    max_generations: int
    seed: int
    crossover_rate: float
    sequence_mutation_rate: float
    factory_mutation_rate: float
    workers: int
    job_clusters: int
    solution_clusters: int
    energy_saving_enabled: bool
    energy_saving_front_limit: int
    energy_saving_passes: int
    energy_saving_start_ratio: float
    energy_saving_interval: int


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


@dataclass
class AdaptiveKMeansSelection:
    best_result: KMeansResult
    best_k: int
    best_score: float
    method: str
    score_trace: list[dict[str, float]]


def _parse_key_value_sections(path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                sections[current_section] = {}
                continue
            if current_section is None or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            sections[current_section][key] = value

    return sections


def _parse_float_pairs(value: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for token in value.split():
        if ":" not in token:
            raise ValueError(f"Invalid machine:value token: {token}")
        key, raw_value = token.split(":", 1)
        pairs.append((key.strip(), float(raw_value)))
    return pairs


def _infer_machines_per_stage(meta: dict[str, str], sections: dict[str, dict[str, str]], stages: int) -> list[int]:
    if "machines_per_stage" in meta:
        values = [int(x) for x in meta["machines_per_stage"].split()]
        if len(values) != stages:
            raise ValueError(f"machines_per_stage length mismatch: expected {stages}, got {len(values)}")
        return values

    machine_layout = sections.get("machine_layout", {})
    if machine_layout:
        inferred = [0] * stages
        for key, value in machine_layout.items():
            if "_stage_" not in key:
                continue
            stage_token = key.split("_stage_", 1)[1]
            stage = int(stage_token) - 1
            inferred[stage] = max(inferred[stage], len(value.split()))
        if all(v > 0 for v in inferred):
            return inferred

    if "machines_per_stage_per_factory" in meta:
        count = int(meta["machines_per_stage_per_factory"])
        return [count] * stages

    raise ValueError("Cannot determine machine-count information in the instance file.")


def parse_instance(path: Path) -> DHFSPInstance:
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    sections = _parse_key_value_sections(path)
    meta = sections.get("meta", {})

    jobs = int(meta["jobs"])
    stages = int(meta["stages"])
    factories = int(meta["factories"])
    name = meta.get("name", path.stem)
    machines_per_stage = _infer_machines_per_stage(meta, sections, stages)
    max_machines = max(machines_per_stage)

    processing_times = np.full((jobs, factories, stages, max_machines), np.nan, dtype=np.float64)

    machine_efficiencies: np.ndarray | None = None
    eff_section = sections.get("machine_efficiencies", {})
    if eff_section:
        machine_efficiencies = np.full((factories, stages, max_machines), np.nan, dtype=np.float64)
        for factory_idx in range(1, factories + 1):
            for stage_idx in range(1, stages + 1):
                key = f"factory_{factory_idx}_stage_{stage_idx}"
                if key not in eff_section:
                    raise ValueError(f"Missing machine efficiencies for {key}")
                pairs = _parse_float_pairs(eff_section[key])
                expected = machines_per_stage[stage_idx - 1]
                if len(pairs) != expected:
                    raise ValueError(
                        f"Invalid machine count for {key}: expected {expected}, got {len(pairs)}"
                    )
                machine_efficiencies[factory_idx - 1, stage_idx - 1, :expected] = [value for _, value in pairs]

    pt_machine_section = sections.get("processing_times_job_factory_stage_machine", {})
    if pt_machine_section:
        for job_idx in range(1, jobs + 1):
            for factory_idx in range(1, factories + 1):
                for stage_idx in range(1, stages + 1):
                    key = f"job_{job_idx:03d}_factory_{factory_idx}_stage_{stage_idx}"
                    if key not in pt_machine_section:
                        raise ValueError(f"Missing processing-times entry for {key}")
                    pairs = _parse_float_pairs(pt_machine_section[key])
                    expected = machines_per_stage[stage_idx - 1]
                    if len(pairs) != expected:
                        raise ValueError(
                            f"Invalid machine count for {key}: expected {expected}, got {len(pairs)}"
                        )
                    processing_times[job_idx - 1, factory_idx - 1, stage_idx - 1, :expected] = [
                        value for _, value in pairs
                    ]
    else:
        pt_legacy_section = sections.get("processing_times_job_factory_stage", {})
        if not pt_legacy_section:
            raise ValueError(
                "Cannot find [processing_times_job_factory_stage_machine] or legacy "
                "[processing_times_job_factory_stage] in the instance file."
            )

        for job_idx in range(1, jobs + 1):
            for factory_idx in range(1, factories + 1):
                key = f"job_{job_idx:03d}_factory_{factory_idx}"
                if key not in pt_legacy_section:
                    raise ValueError(f"Missing legacy processing-times entry for {key}")
                values = [float(x) for x in pt_legacy_section[key].split()]
                if len(values) != stages:
                    raise ValueError(f"Invalid stage count for {key}: expected {stages}, got {len(values)}")
                for stage_idx, value in enumerate(values):
                    m_count = machines_per_stage[stage_idx]
                    processing_times[job_idx - 1, factory_idx - 1, stage_idx, :m_count] = value

    due_dates = np.zeros(jobs, dtype=np.float64)
    due_section = sections.get("due_dates", {})
    for job_idx in range(1, jobs + 1):
        key = f"job_{job_idx:03d}"
        if key not in due_section:
            raise ValueError(f"Missing due date for {key}")
        due_dates[job_idx - 1] = float(due_section[key])

    workloads: np.ndarray | None = None
    workload_section = sections.get("job_workloads", {})
    if workload_section:
        workloads = np.zeros(jobs, dtype=np.float64)
        for job_idx in range(1, jobs + 1):
            key = f"job_{job_idx:03d}"
            if key not in workload_section:
                raise ValueError(f"Missing workload for {key}")
            workloads[job_idx - 1] = float(workload_section[key])

    return DHFSPInstance(
        name=name,
        jobs=jobs,
        stages=stages,
        factories=factories,
        machines_per_stage=machines_per_stage,
        processing_times=processing_times,
        due_dates=due_dates,
        workloads=workloads,
        machine_efficiencies=machine_efficiencies,
        source_path=path,
    )


def dominates(a: Solution, b: Solution) -> bool:
    assert a.objectives is not None and b.objectives is not None
    return (
        a.objectives[0] <= b.objectives[0]
        and a.objectives[1] <= b.objectives[1]
        and (a.objectives[0] < b.objectives[0] or a.objectives[1] < b.objectives[1])
    )


def crowded_better(a: Solution, b: Solution) -> bool:
    if a.rank != b.rank:
        return a.rank < b.rank
    return a.crowding_distance > b.crowding_distance


def _safe_standardize(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("Expected a 2D array for standardization.")
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (data - mean) / std


def _pairwise_squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = data[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _kmeans_plus_plus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=np.float64)
    first_idx = int(rng.integers(0, n_samples))
    centers[0] = data[first_idx]

    closest_sq = np.sum((data - centers[0]) ** 2, axis=1)
    for center_idx in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 1e-12:
            candidate = int(rng.integers(0, n_samples))
        else:
            probs = closest_sq / total
            candidate = int(rng.choice(n_samples, p=probs))
        centers[center_idx] = data[candidate]
        new_sq = np.sum((data - centers[center_idx]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def run_kmeans(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    n_init: int = 6,
    max_iter: int = 50,
) -> KMeansResult:
    if data.ndim != 2:
        raise ValueError("run_kmeans expects a 2D array.")
    n_samples = data.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot cluster an empty dataset.")
    k = max(1, min(int(k), n_samples))
    if k == 1:
        center = np.mean(data, axis=0, keepdims=True)
        inertia = float(np.sum((data - center) ** 2))
        return KMeansResult(labels=np.zeros(n_samples, dtype=int), centers=center, inertia=inertia)

    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")

    for _ in range(max(1, n_init)):
        centers = _kmeans_plus_plus_init(data, k, rng)
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max(1, max_iter)):
            distances = _pairwise_squared_distances(data, centers)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

            new_centers = centers.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centers[cluster_id] = np.mean(data[mask], axis=0)
                else:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    new_centers[cluster_id] = data[farthest]
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        final_distances = _pairwise_squared_distances(data, centers)
        labels = np.argmin(final_distances, axis=1)
        inertia = float(np.sum(final_distances[np.arange(n_samples), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    assert best_labels is not None and best_centers is not None
    return KMeansResult(labels=best_labels.astype(int), centers=best_centers, inertia=best_inertia)



def _pairwise_sample_distances(data: np.ndarray) -> np.ndarray:
    diff = data[:, None, :] - data[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def compute_silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    if data.ndim != 2:
        raise ValueError("compute_silhouette_score expects a 2D array.")
    n_samples = data.shape[0]
    if n_samples <= 2:
        return 0.0

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return 0.0

    distances = _pairwise_sample_distances(data)
    silhouettes = np.zeros(n_samples, dtype=np.float64)

    for idx in range(n_samples):
        label = labels[idx]
        same_mask = labels == label
        same_count = int(np.sum(same_mask))
        if same_count <= 1:
            silhouettes[idx] = 0.0
            continue

        a = float(np.sum(distances[idx, same_mask]) / max(1, same_count - 1))
        b = float("inf")
        for other_label in unique_labels.tolist():
            if int(other_label) == int(label):
                continue
            other_mask = labels == other_label
            if not np.any(other_mask):
                continue
            mean_dist = float(np.mean(distances[idx, other_mask]))
            if mean_dist < b:
                b = mean_dist

        if not np.isfinite(b):
            silhouettes[idx] = 0.0
            continue
        denom = max(a, b, 1e-12)
        silhouettes[idx] = (b - a) / denom

    return float(np.mean(silhouettes))


def select_adaptive_kmeans(
    data: np.ndarray,
    rng: np.random.Generator,
    *,
    min_k: int = 2,
    max_k: int = 6,
    n_init: int = 6,
    max_iter: int = 50,
    prefer_smaller_k_margin: float = 0.02,
) -> AdaptiveKMeansSelection:
    if data.ndim != 2:
        raise ValueError("select_adaptive_kmeans expects a 2D array.")
    n_samples = data.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot cluster an empty dataset.")
    if n_samples <= 2:
        result = run_kmeans(data, 1, rng, n_init=1, max_iter=max_iter)
        return AdaptiveKMeansSelection(
            best_result=result,
            best_k=1,
            best_score=0.0,
            method="degenerate_single_cluster",
            score_trace=[{"k": 1.0, "silhouette": 0.0, "inertia": float(result.inertia)}],
        )

    lower = max(2, min_k)
    upper = min(max_k, n_samples - 1)
    if upper < lower:
        result = run_kmeans(data, 1, rng, n_init=1, max_iter=max_iter)
        return AdaptiveKMeansSelection(
            best_result=result,
            best_k=1,
            best_score=0.0,
            method="fallback_single_cluster",
            score_trace=[{"k": 1.0, "silhouette": 0.0, "inertia": float(result.inertia)}],
        )

    trace: list[dict[str, float]] = []
    best_selection: tuple[float, int, float, KMeansResult] | None = None
    best_score = -float("inf")

    for k in range(lower, upper + 1):
        result = run_kmeans(data, k, rng, n_init=n_init, max_iter=max_iter)
        score = compute_silhouette_score(data, result.labels)
        trace.append({"k": float(k), "silhouette": float(score), "inertia": float(result.inertia)})
        if best_selection is None:
            best_selection = (score, k, result.inertia, result)
            best_score = score
            continue
        if score > best_score + 1e-12:
            best_selection = (score, k, result.inertia, result)
            best_score = score
        elif abs(score - best_score) <= prefer_smaller_k_margin:
            assert best_selection is not None
            _, best_k_now, best_inertia_now, _ = best_selection
            if k < best_k_now or (k == best_k_now and result.inertia < best_inertia_now):
                best_selection = (score, k, result.inertia, result)

    assert best_selection is not None
    score, best_k, _, best_result = best_selection
    return AdaptiveKMeansSelection(
        best_result=best_result,
        best_k=int(best_k),
        best_score=float(score),
        method="silhouette_kmeans",
        score_trace=trace,
    )


_WORKER_INSTANCE: DHFSPInstance | None = None


def _worker_init(instance: DHFSPInstance) -> None:
    global _WORKER_INSTANCE
    _WORKER_INSTANCE = instance


def _evaluate_solution_core(
    instance: DHFSPInstance,
    permutation: np.ndarray,
    factory_assignment: np.ndarray,
) -> tuple[tuple[float, float], np.ndarray, float, float]:
    jobs = instance.jobs
    factories = instance.factories
    stages = instance.stages
    max_machines = max(instance.machines_per_stage)

    machine_ready = np.zeros((factories, stages, max_machines), dtype=np.float64)
    machine_first_start = np.full((factories, stages, max_machines), np.inf, dtype=np.float64)
    machine_last_finish = np.zeros((factories, stages, max_machines), dtype=np.float64)
    machine_used = np.zeros((factories, stages, max_machines), dtype=bool)
    completion_times = np.zeros(jobs, dtype=np.float64)

    total_processing = 0.0

    for job in permutation.tolist():
        factory = int(factory_assignment[job])
        prev_finish = 0.0

        for stage in range(stages):
            m_count = instance.machines_per_stage[stage]
            ready = machine_ready[factory, stage, :m_count]
            stage_processing = instance.processing_times[job, factory, stage, :m_count]

            starts = np.maximum(ready, prev_finish)
            finishes = starts + stage_processing
            machine = int(np.argmin(finishes))

            start = starts[machine]
            finish = finishes[machine]

            machine_ready[factory, stage, machine] = finish
            machine_last_finish[factory, stage, machine] = finish
            if not machine_used[factory, stage, machine]:
                machine_first_start[factory, stage, machine] = start
                machine_used[factory, stage, machine] = True

            prev_finish = finish
            total_processing += float(stage_processing[machine])

        completion_times[job] = prev_finish

    tardiness = np.maximum(completion_times - instance.due_dates, 0.0)
    ttd = float(np.sum(tardiness))

    if np.any(machine_used):
        on_time = float(np.sum(machine_last_finish[machine_used] - machine_first_start[machine_used]))
    else:
        on_time = 0.0

    tec = float(on_time + 4.0 * total_processing)
    return (ttd, tec), completion_times, on_time, total_processing


def _evaluate_solution_worker(
    task: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[float, float], np.ndarray, float, float]:
    if _WORKER_INSTANCE is None:
        raise RuntimeError("Worker instance is not initialized.")
    permutation, factory_assignment = task
    return _evaluate_solution_core(_WORKER_INSTANCE, permutation, factory_assignment)


class NSGA2DHFSPRouteC:
    def __init__(
        self,
        instance: DHFSPInstance,
        population_size: int,
        seed: int,
        max_generations: int = 400,
        crossover_rate: float = 0.9,
        sequence_mutation_rate: float = 0.2,
        factory_mutation_rate: float = 0.1,
        workers: int = 1,
        job_clusters: int = 0,
        solution_clusters: int = 0,
        energy_saving_enabled: bool = True,
        energy_saving_front_limit: int = 10,
        energy_saving_passes: int = 1,
        energy_saving_start_ratio: float = 0.5,
        energy_saving_interval: int = 5,
        energy_saving_max_trials_per_solution: int | None = None,
        energy_saving_max_trials_per_call: int | None = None,
    ) -> None:
        self.instance = instance
        self.population_size = max(4, population_size)
        self.rng = np.random.default_rng(seed)
        self.crossover_rate = crossover_rate
        self.sequence_mutation_rate = sequence_mutation_rate
        self.factory_mutation_rate = factory_mutation_rate
        self.nefs = 0
        self.max_generations = max(1, int(max_generations))

        cpu_total = os.cpu_count() or 1
        self.workers = max(1, min(int(workers), cpu_total))
        self._executor: ProcessPoolExecutor | None = None

        self.mean_job_factory_processing = self._aggregate_job_factory_processing(mode="mean")
        self.best_job_factory_processing = self._aggregate_job_factory_processing(mode="min")
        self.avg_job_work = self.mean_job_factory_processing.mean(axis=1)
        self.best_job_work = np.min(self.best_job_factory_processing, axis=1)
        self.stage_mean_profile = self._compute_stage_mean_profile()
        self.preference_gap = self._compute_preference_gap()
        self.factory_sensitivity = np.std(self.mean_job_factory_processing, axis=1)
        self.job_urgency_score = self._compute_job_urgency_score()

        self.job_feature_matrix = self._build_job_feature_matrix()
        if job_clusters > 0:
            self.job_cluster_count = max(1, min(int(job_clusters), self.instance.jobs))
            self.job_cluster_result = run_kmeans(self.job_feature_matrix, self.job_cluster_count, self.rng)
            self.job_cluster_selection = AdaptiveKMeansSelection(
                best_result=self.job_cluster_result,
                best_k=self.job_cluster_count,
                best_score=float('nan'),
                method="manual_kmeans",
                score_trace=[{"k": float(self.job_cluster_count), "silhouette": float('nan'), "inertia": float(self.job_cluster_result.inertia)}],
            )
        else:
            job_k_upper = self._auto_job_cluster_upper_bound()
            self.job_cluster_selection = select_adaptive_kmeans(
                self.job_feature_matrix,
                self.rng,
                min_k=2,
                max_k=job_k_upper,
                n_init=8,
                max_iter=60,
                prefer_smaller_k_margin=0.02,
            )
            self.job_cluster_result = self.job_cluster_selection.best_result
            self.job_cluster_count = self.job_cluster_selection.best_k
        self.job_cluster_labels = self.job_cluster_result.labels
        self.job_cluster_sizes = np.bincount(self.job_cluster_labels, minlength=self.job_cluster_count).astype(int)
        self.cluster_factory_preferences = self._build_cluster_factory_preferences()
        self.cluster_factory_rank_penalties = self._build_cluster_factory_rank_penalties()
        self.cluster_profiles = self._build_cluster_profiles()

        self.solution_cluster_count = max(0, int(solution_clusters))
        self.energy_saving_enabled = bool(energy_saving_enabled)
        self.energy_saving_front_limit = int(energy_saving_front_limit)
        self.energy_saving_passes = max(1, int(energy_saving_passes))
        self.energy_saving_start_ratio = min(1.0, max(0.0, float(energy_saving_start_ratio)))
        self.energy_saving_interval = max(1, int(energy_saving_interval))


        self.energy_saving_trials_mode = "auto"
        self.energy_saving_max_trials_per_solution = self._auto_energy_saving_trials_per_solution()
        self.energy_saving_max_trials_per_call = self._auto_energy_saving_trials_per_call()

    def _aggregate_job_factory_processing(self, mode: str) -> np.ndarray:
        result = np.zeros((self.instance.jobs, self.instance.factories), dtype=np.float64)
        for stage, m_count in enumerate(self.instance.machines_per_stage):
            block = self.instance.processing_times[:, :, stage, :m_count]
            if mode == "mean":
                result += np.mean(block, axis=2)
            elif mode == "min":
                result += np.min(block, axis=2)
            else:
                raise ValueError(f"Unsupported aggregation mode: {mode}")
        return result

    def _compute_stage_mean_profile(self) -> np.ndarray:
        profile = np.zeros((self.instance.jobs, self.instance.stages), dtype=np.float64)
        for stage, m_count in enumerate(self.instance.machines_per_stage):
            block = self.instance.processing_times[:, :, stage, :m_count]
            profile[:, stage] = np.mean(block, axis=(1, 2))
        return profile

    def _compute_preference_gap(self) -> np.ndarray:
        if self.instance.factories == 1:
            return np.zeros(self.instance.jobs, dtype=np.float64)
        ordered = np.sort(self.best_job_factory_processing, axis=1)
        denom = np.maximum(ordered[:, 0], 1e-9)
        return (ordered[:, 1] - ordered[:, 0]) / denom

    def _compute_job_urgency_score(self) -> np.ndarray:
        slack = self.instance.due_dates - self.best_job_work
        base = np.column_stack([self.instance.due_dates, self.best_job_work, slack])
        scaled = _safe_standardize(base)
        urgency = 0.65 * (-scaled[:, 0]) + 0.35 * scaled[:, 1] + 0.45 * (-scaled[:, 2])
        urgency -= np.min(urgency)
        denom = float(np.max(urgency))
        if denom <= 1e-12:
            return np.zeros_like(urgency)
        return urgency / denom

    def _build_job_feature_matrix(self) -> np.ndarray:
        slack_ratio = (self.instance.due_dates - self.best_job_work) / np.maximum(self.best_job_work, 1e-9)
        dominant_stage_ratio = np.max(self.stage_mean_profile, axis=1) / np.maximum(
            np.sum(self.stage_mean_profile, axis=1),
            1e-9,
        )
        features = [
            self.instance.due_dates,
            self.avg_job_work,
            self.best_job_work,
            slack_ratio,
            self.preference_gap,
            self.factory_sensitivity,
            dominant_stage_ratio,
        ]
        if self.instance.workloads is not None:
            features.append(self.instance.workloads)
        matrix = np.column_stack(features).astype(np.float64)
        return _safe_standardize(matrix)

    def _auto_job_cluster_upper_bound(self) -> int:
        if self.instance.jobs <= 2:
            return 1
        guess = int(round(math.sqrt(self.instance.jobs))) + 2
        return max(2, min(8, guess))

    def _auto_solution_cluster_upper_bound(self, front_size: int, remaining: int) -> int:
        upper = min(front_size - 1, remaining, 8)
        if upper <= 1:
            return 1
        heuristic = max(2, int(round(math.sqrt(max(4, min(front_size, self.population_size))))))
        return max(2, min(upper, heuristic + 1))

    def _auto_energy_saving_trials_per_solution(self) -> int:
        return max(1, self.instance.jobs // 2)

    def _auto_energy_saving_trials_per_call(self) -> int:
        effective_front_limit = (
            self.energy_saving_front_limit
            if self.energy_saving_front_limit > 0
            else self.population_size
        )
        return max(1, effective_front_limit) * self.energy_saving_max_trials_per_solution

    def _build_cluster_factory_preferences(self) -> np.ndarray:
        preferences = np.zeros((self.job_cluster_count, self.instance.factories), dtype=int)
        for cluster_id in range(self.job_cluster_count):
            members = np.where(self.job_cluster_labels == cluster_id)[0]
            if members.size == 0:
                preferences[cluster_id] = np.arange(self.instance.factories, dtype=int)
                continue
            cluster_cost = np.mean(self.best_job_factory_processing[members], axis=0)
            preferences[cluster_id] = np.argsort(cluster_cost, kind="stable")
        return preferences

    def _build_cluster_factory_rank_penalties(self) -> np.ndarray:
        penalties = np.zeros((self.job_cluster_count, self.instance.factories), dtype=np.float64)
        for cluster_id in range(self.job_cluster_count):
            for rank, factory in enumerate(self.cluster_factory_preferences[cluster_id].tolist()):
                penalties[cluster_id, factory] = float(rank)
        denom = max(1, self.instance.factories - 1)
        return penalties / denom

    def _build_cluster_profiles(self) -> dict[int, dict[str, float]]:
        profiles: dict[int, dict[str, float]] = {}
        slack_ratio = (self.instance.due_dates - self.best_job_work) / np.maximum(self.best_job_work, 1e-9)
        for cluster_id in range(self.job_cluster_count):
            members = np.where(self.job_cluster_labels == cluster_id)[0]
            if members.size == 0:
                profiles[cluster_id] = {
                    "mean_due": 0.0,
                    "mean_work": 0.0,
                    "mean_pref_gap": 0.0,
                    "mean_slack_ratio": 0.0,
                    "mean_urgency": 0.0,
                }
                continue
            profiles[cluster_id] = {
                "mean_due": float(np.mean(self.instance.due_dates[members])),
                "mean_work": float(np.mean(self.avg_job_work[members])),
                "mean_pref_gap": float(np.mean(self.preference_gap[members])),
                "mean_slack_ratio": float(np.mean(slack_ratio[members])),
                "mean_urgency": float(np.mean(self.job_urgency_score[members])),
            }
        return profiles

    def canonicalize_permutation(self, permutation: np.ndarray, factory_assignment: np.ndarray) -> np.ndarray:
        blocks = self.extract_factory_blocks(permutation, factory_assignment)
        return self.flatten_factory_blocks(blocks)

    def extract_factory_blocks(
        self,
        permutation: np.ndarray,
        factory_assignment: np.ndarray,
    ) -> list[list[int]]:
        blocks: list[list[int]] = [[] for _ in range(self.instance.factories)]
        for job in permutation.tolist():
            factory = int(factory_assignment[job])
            blocks[factory].append(int(job))

        total_jobs = sum(len(block) for block in blocks)
        if total_jobs != self.instance.jobs:
            raise ValueError(
                f"Invalid encoding: expected {self.instance.jobs} jobs across factory blocks, got {total_jobs}."
            )
        return blocks

    def flatten_factory_blocks(self, blocks: list[list[int]]) -> np.ndarray:
        flattened = [job for block in blocks for job in block]
        if len(flattened) != self.instance.jobs:
            raise ValueError(
                f"Invalid flattened encoding: expected {self.instance.jobs} jobs, got {len(flattened)}."
            )
        return np.array(flattened, dtype=int)

    @staticmethod
    def invalidate_solution(solution: Solution) -> None:
        solution.objectives = None
        solution.completion_times = None
        solution.machine_on_time = None
        solution.processing_time_total = None

    def canonicalize_solution(self, solution: Solution) -> None:
        solution.permutation = self.canonicalize_permutation(solution.permutation, solution.factory_assignment)
        self.invalidate_solution(solution)


    @staticmethod
    def _assign_evaluation_to_solution(
        solution: Solution,
        evaluation: tuple[tuple[float, float], np.ndarray, float, float],
    ) -> None:
        objectives, completion_times, on_time, total_processing = evaluation
        solution.objectives = objectives
        solution.completion_times = completion_times
        solution.machine_on_time = on_time
        solution.processing_time_total = total_processing

    def _evaluate_candidate(
        self,
        permutation: np.ndarray,
        factory_assignment: np.ndarray,
        *,
        count_nef: bool = True,
    ) -> tuple[tuple[float, float], np.ndarray, float, float]:
        evaluation = _evaluate_solution_core(self.instance, permutation, factory_assignment)
        if count_nef:
            self.nefs += 1
        return evaluation

    @staticmethod
    def _move_job_in_block(block: list[int], from_idx: int, to_idx: int) -> list[int]:
        moved = block.copy()
        job = moved.pop(from_idx)
        adjusted_to = to_idx
        if adjusted_to > from_idx:
            adjusted_to -= 1
        moved.insert(adjusted_to, job)
        return moved

    @staticmethod
    def _energy_saving_accepts(
        candidate: tuple[float, float],
        incumbent: tuple[float, float],
        eps: float = 1e-9,
    ) -> bool:
        candidate_ttd, candidate_tec = candidate
        incumbent_ttd, incumbent_tec = incumbent
        if candidate_tec < incumbent_tec - eps and candidate_ttd <= incumbent_ttd + eps:
            return True
        if abs(candidate_tec - incumbent_tec) <= eps and candidate_ttd < incumbent_ttd - eps:
            return True
        return False

    def energy_saving_delay_ready(self, generation: int) -> bool:
        if not self.energy_saving_enabled:
            return False
        if self.energy_saving_start_ratio <= 0.0:
            return True
        trigger_generation = int(math.ceil(self.energy_saving_start_ratio * self.max_generations))
        return generation >= trigger_generation

    def should_trigger_energy_saving(self, generation: int, *, is_final: bool = False) -> bool:
        if not self.energy_saving_enabled:
            return False
        if not self.energy_saving_delay_ready(generation):
            return False
        if is_final:
            return True
        return generation > 0 and generation % self.energy_saving_interval == 0

    def energy_save_solution(self, solution: Solution, *, trial_limit: int | None = None) -> tuple[bool, int]:
        if not self.energy_saving_enabled:
            return False, 0

        max_trials = (
            self.energy_saving_max_trials_per_solution
            if trial_limit is None
            else max(0, int(trial_limit))
        )
        if max_trials <= 0:
            return False, 0

        solution.permutation = self.canonicalize_permutation(solution.permutation, solution.factory_assignment)

        trials_used = 0
        if solution.objectives is None:
            evaluation = self._evaluate_candidate(solution.permutation, solution.factory_assignment, count_nef=True)
            self._assign_evaluation_to_solution(solution, evaluation)
            trials_used += 1
            if trials_used >= max_trials:
                return False, trials_used

        improved_any = False
        for _ in range(self.energy_saving_passes):
            if trials_used >= max_trials:
                break
            pass_improved = False
            for direction in ("forward", "backward"):
                if trials_used >= max_trials:
                    break
                while trials_used < max_trials:
                    blocks = self.extract_factory_blocks(solution.permutation, solution.factory_assignment)
                    accepted_move = False
                    budget_exhausted = False

                    for factory in range(self.instance.factories):
                        block = blocks[factory]
                        if len(block) <= 1:
                            continue

                        if direction == "forward":
                            idx_iter = range(1, len(block))
                        else:
                            idx_iter = range(len(block) - 2, -1, -1)

                        for idx in idx_iter:
                            if direction == "forward":
                                pos_iter = range(0, idx)
                            else:
                                pos_iter = range(idx + 1, len(block) + 1)

                            for pos in pos_iter:
                                if trials_used >= max_trials:
                                    budget_exhausted = True
                                    break

                                moved_block = self._move_job_in_block(block, idx, pos)
                                if moved_block == block:
                                    continue

                                candidate_blocks = [list(factory_block) for factory_block in blocks]
                                candidate_blocks[factory] = moved_block
                                candidate_permutation = self.flatten_factory_blocks(candidate_blocks)
                                evaluation = self._evaluate_candidate(
                                    candidate_permutation,
                                    solution.factory_assignment,
                                    count_nef=True,
                                )

                                trials_used += 1
                                if self._energy_saving_accepts(evaluation[0], solution.objectives):
                                    solution.permutation = candidate_permutation
                                    self._assign_evaluation_to_solution(solution, evaluation)
                                    improved_any = True
                                    pass_improved = True
                                    accepted_move = True
                                    break
                            if accepted_move or budget_exhausted:
                                break
                        if accepted_move or budget_exhausted:
                            break

                    if budget_exhausted or not accepted_move:
                        break

            if not pass_improved:
                break

        return improved_any, trials_used

    def apply_energy_saving_to_population(
        self,
        population: list[Solution],
        *,
        generation: int,
        is_final: bool = False,
    ) -> int:
        if not population:
            return 0
        if not self.should_trigger_energy_saving(generation, is_final=is_final):
            return 0

        fronts = self.fast_non_dominated_sort(population)
        self.assign_crowding_distance(fronts)
        elite_front = list(fronts[0]) if fronts else population
        elite_front.sort(
            key=lambda sol: (sol.crowding_distance, -sol.objectives[0], -sol.objectives[1]),
            reverse=True,
        )

        if self.energy_saving_front_limit <= 0:
            limit = len(elite_front)
        else:
            limit = min(len(elite_front), self.energy_saving_front_limit)

        trials_used = 0
        call_limit = max(1, self.energy_saving_max_trials_per_call)

        for solution in elite_front[:limit]:
            if trials_used >= call_limit:
                break
            remaining_trials = call_limit - trials_used
            _, used = self.energy_save_solution(solution, trial_limit=remaining_trials)
            trials_used += used

        return trials_used

    @contextmanager
    def executor_scope(self):
        if self.workers <= 1:
            self._executor = None
            yield
            return

        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_worker_init,
            initargs=(self.instance,),
        ) as executor:
            self._executor = executor
            yield
            self._executor = None

    def solve(self) -> tuple[list[Solution], list[dict[str, float]]]:
        history: list[dict[str, float]] = []

        with self.executor_scope():
            population = self.initialize_population(self.population_size)
            self.evaluate_population(population)

            generation = 0
            while generation < self.max_generations:
                fronts = self.fast_non_dominated_sort(population)
                self.assign_crowding_distance(fronts)
                history.append(self.snapshot(generation, population, fronts))

                offspring_target = self.population_size

                offspring: list[Solution] = []
                while len(offspring) < offspring_target:
                    p1 = self.tournament_selection(population)
                    p2 = self.tournament_selection(population)
                    c1, c2 = self.crossover(p1, p2)
                    self.mutate(c1)
                    self.mutate(c2)
                    offspring.append(c1)
                    if len(offspring) < offspring_target:
                        offspring.append(c2)

                self.evaluate_population(offspring)

                combined = population + offspring
                combined_fronts = self.fast_non_dominated_sort(combined)
                self.assign_crowding_distance(combined_fronts)
                population = self.environmental_selection(combined_fronts, self.population_size)
                generation += 1
                self.apply_energy_saving_to_population(population, generation=generation)

            self.apply_energy_saving_to_population(population, generation=generation, is_final=True)
            final_fronts = self.fast_non_dominated_sort(population)
            self.assign_crowding_distance(final_fronts)
            history.append(self.snapshot(generation, population, final_fronts))
            pareto = [sol.clone() for sol in final_fronts[0]]
            pareto.sort(key=lambda s: (s.objectives[0], s.objectives[1]))

        return pareto, history

    def initialize_population(self, size: int) -> list[Solution]:
        population: list[Solution] = []

        heuristic_builders = [
            self.make_cluster_due_balanced_solution,
            self.make_cluster_workload_balanced_solution,
            self.make_cluster_preference_solution,
            self.make_cluster_mixed_solution,
            self.make_edd_balanced_solution,
            self.make_spt_balanced_solution,
            self.make_lpt_balanced_solution,
            self.make_energy_biased_solution,
        ]
        for builder in heuristic_builders:
            if len(population) >= size:
                break
            solution = builder()
            self.canonicalize_solution(solution)
            population.append(solution)

        while len(population) < size:
            permutation = self.make_random_cluster_aware_permutation()
            factory_assignment = self.cluster_guided_assignment(permutation, strategy="mixed", stochastic=True)
            solution = Solution(permutation=permutation, factory_assignment=factory_assignment)
            self.canonicalize_solution(solution)
            population.append(solution)
        return population

    def make_edd_balanced_solution(self) -> Solution:
        order = np.argsort(self.instance.due_dates, kind="stable")
        assignment = self.greedy_balance_assignment(order)
        return Solution(permutation=order.astype(int), factory_assignment=assignment)

    def make_spt_balanced_solution(self) -> Solution:
        order = np.argsort(self.avg_job_work, kind="stable")
        assignment = self.greedy_balance_assignment(order)
        return Solution(permutation=order.astype(int), factory_assignment=assignment)

    def make_lpt_balanced_solution(self) -> Solution:
        order = np.argsort(-self.avg_job_work, kind="stable")
        assignment = self.greedy_balance_assignment(order)
        return Solution(permutation=order.astype(int), factory_assignment=assignment)

    def make_energy_biased_solution(self) -> Solution:
        order = np.argsort(self.instance.due_dates + 0.2 * self.avg_job_work, kind="stable")
        assignment = np.argmin(self.best_job_factory_processing, axis=1)
        return Solution(permutation=order.astype(int), factory_assignment=assignment.astype(int))

    def make_cluster_due_balanced_solution(self) -> Solution:
        order = self.build_cluster_order(strategy="due")
        assignment = self.cluster_guided_assignment(order, strategy="due")
        return Solution(permutation=order, factory_assignment=assignment)

    def make_cluster_workload_balanced_solution(self) -> Solution:
        order = self.build_cluster_order(strategy="work")
        assignment = self.cluster_guided_assignment(order, strategy="work")
        return Solution(permutation=order, factory_assignment=assignment)

    def make_cluster_preference_solution(self) -> Solution:
        order = self.build_cluster_order(strategy="pref")
        assignment = self.cluster_guided_assignment(order, strategy="pref")
        return Solution(permutation=order, factory_assignment=assignment)

    def make_cluster_mixed_solution(self) -> Solution:
        order = self.build_cluster_order(strategy="mixed")
        assignment = self.cluster_guided_assignment(order, strategy="mixed")
        return Solution(permutation=order, factory_assignment=assignment)

    def build_cluster_order(self, strategy: str) -> np.ndarray:
        cluster_ids = list(range(self.job_cluster_count))
        if strategy == "due":
            cluster_ids.sort(key=lambda cid: (self.cluster_profiles[cid]["mean_due"], -self.cluster_profiles[cid]["mean_work"]))
        elif strategy == "work":
            cluster_ids.sort(key=lambda cid: (-self.cluster_profiles[cid]["mean_work"], self.cluster_profiles[cid]["mean_due"]))
        elif strategy == "pref":
            cluster_ids.sort(key=lambda cid: (-self.cluster_profiles[cid]["mean_pref_gap"], -self.cluster_profiles[cid]["mean_urgency"]))
        else:
            cluster_ids.sort(
                key=lambda cid: (
                    -self.cluster_profiles[cid]["mean_urgency"],
                    self.cluster_profiles[cid]["mean_due"],
                    -self.cluster_profiles[cid]["mean_pref_gap"],
                )
            )

        ordered_jobs: list[int] = []
        for cluster_id in cluster_ids:
            members = np.where(self.job_cluster_labels == cluster_id)[0].tolist()
            if strategy == "due":
                members.sort(key=lambda job: (self.instance.due_dates[job], self.avg_job_work[job]))
            elif strategy == "work":
                members.sort(key=lambda job: (-self.avg_job_work[job], self.instance.due_dates[job]))
            elif strategy == "pref":
                members.sort(key=lambda job: (-self.preference_gap[job], self.instance.due_dates[job]))
            else:
                members.sort(
                    key=lambda job: (
                        -self.job_urgency_score[job],
                        self.instance.due_dates[job],
                        -self.preference_gap[job],
                    )
                )
            ordered_jobs.extend(members)
        return np.array(ordered_jobs, dtype=int)

    def make_random_cluster_aware_permutation(self) -> np.ndarray:
        cluster_ids = list(range(self.job_cluster_count))
        self.rng.shuffle(cluster_ids)
        permutation: list[int] = []
        for cluster_id in cluster_ids:
            members = np.where(self.job_cluster_labels == cluster_id)[0].tolist()
            members.sort(key=lambda job: (self.instance.due_dates[job], -self.avg_job_work[job]))
            if len(members) > 1:
                cut = int(self.rng.integers(0, len(members)))
                members = members[cut:] + members[:cut]
            permutation.extend(members)
        return np.array(permutation, dtype=int)

    def greedy_balance_assignment(self, order: np.ndarray) -> np.ndarray:
        loads = np.zeros(self.instance.factories, dtype=np.float64)
        assignment = np.zeros(self.instance.jobs, dtype=int)

        for job in order.tolist():
            candidate_loads = loads + self.mean_job_factory_processing[job]
            factory = int(np.argmin(candidate_loads))
            assignment[job] = factory
            loads[factory] += self.mean_job_factory_processing[job, factory]

        return assignment

    def cluster_guided_assignment(self, order: np.ndarray, strategy: str, stochastic: bool = False) -> np.ndarray:
        assignment = np.zeros(self.instance.jobs, dtype=int)
        loads = np.zeros(self.instance.factories, dtype=np.float64)
        cluster_counts = np.zeros((self.job_cluster_count, self.instance.factories), dtype=np.int64)

        for job in order.tolist():
            cluster_id = int(self.job_cluster_labels[job])
            scores = np.zeros(self.instance.factories, dtype=np.float64)
            scale = max(1.0, self.avg_job_work[job])
            urgency = self.job_urgency_score[job]
            for factory in range(self.instance.factories):
                pref_penalty = self.cluster_factory_rank_penalties[cluster_id, factory] * (0.18 * scale)
                crowd_penalty = cluster_counts[cluster_id, factory] * (0.10 * scale)
                load_term = loads[factory] + self.mean_job_factory_processing[job, factory]
                proc_term = 0.55 * self.mean_job_factory_processing[job, factory] + (0.25 + 0.15 * urgency) * self.best_job_factory_processing[job, factory]
                score = load_term + proc_term + pref_penalty + crowd_penalty
                if strategy == "due":
                    score += urgency * 0.40 * self.best_job_factory_processing[job, factory]
                elif strategy == "work":
                    score += 0.12 * self.mean_job_factory_processing[job, factory]
                elif strategy == "pref":
                    score += self.cluster_factory_rank_penalties[cluster_id, factory] * (0.30 * scale)
                else:
                    score += urgency * 0.22 * self.best_job_factory_processing[job, factory]
                scores[factory] = score

            if stochastic and self.instance.factories > 1 and self.rng.random() < 0.35:
                ranked = np.argsort(scores)
                top_k = min(2, len(ranked))
                choice = int(self.rng.choice(ranked[:top_k]))
                factory = choice
            else:
                factory = int(np.argmin(scores))

            assignment[job] = factory
            loads[factory] += self.mean_job_factory_processing[job, factory]
            cluster_counts[cluster_id, factory] += 1

        return assignment

    def evaluate_population(self, population: Iterable[Solution]) -> None:
        population = list(population)
        if not population:
            return

        for solution in population:
            solution.permutation = self.canonicalize_permutation(solution.permutation, solution.factory_assignment)

        if self._executor is None or self.workers <= 1 or len(population) == 1:
            for solution in population:
                objectives, completion_times, on_time, total_processing = _evaluate_solution_core(
                    self.instance,
                    solution.permutation,
                    solution.factory_assignment,
                )
                self._assign_evaluation_to_solution(solution, (objectives, completion_times, on_time, total_processing))
        else:
            tasks = [(solution.permutation.copy(), solution.factory_assignment.copy()) for solution in population]
            chunksize = max(1, len(tasks) // (self.workers * 4))
            results = self._executor.map(_evaluate_solution_worker, tasks, chunksize=chunksize)

            for solution, result in zip(population, results):
                objectives, completion_times, on_time, total_processing = result
                self._assign_evaluation_to_solution(solution, (objectives, completion_times, on_time, total_processing))

        self.nefs += len(population)

    def crossover(self, parent1: Solution, parent2: Solution) -> tuple[Solution, Solution]:
        c1 = parent1.clone()
        c2 = parent2.clone()

        if self.rng.random() < self.crossover_rate:
            c1.permutation, c2.permutation = self.order_crossover(parent1.permutation, parent2.permutation)
            mask = self.rng.random(self.instance.jobs) < 0.5
            c1.factory_assignment = np.where(mask, parent1.factory_assignment, parent2.factory_assignment).astype(int)
            c2.factory_assignment = np.where(mask, parent2.factory_assignment, parent1.factory_assignment).astype(int)
        else:
            c1.permutation = parent1.permutation.copy()
            c2.permutation = parent2.permutation.copy()
            c1.factory_assignment = parent1.factory_assignment.copy()
            c2.factory_assignment = parent2.factory_assignment.copy()

        self.canonicalize_solution(c1)
        self.canonicalize_solution(c2)
        return c1, c2

    def order_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(p1)
        a, b = sorted(self.rng.choice(n, size=2, replace=False).tolist())

        child1 = np.full(n, -1, dtype=int)
        child2 = np.full(n, -1, dtype=int)
        child1[a : b + 1] = p1[a : b + 1]
        child2[a : b + 1] = p2[a : b + 1]

        self.fill_ordered(child1, p2, b + 1)
        self.fill_ordered(child2, p1, b + 1)
        return child1, child2

    @staticmethod
    def fill_ordered(child: np.ndarray, donor: np.ndarray, start_idx: int) -> None:
        n = len(child)
        present = np.zeros(n, dtype=bool)
        fixed = child[child != -1]
        if fixed.size > 0:
            present[fixed] = True

        insert_pos = start_idx % n
        for gene in donor.tolist():
            if present[gene]:
                continue
            while child[insert_pos] != -1:
                insert_pos = (insert_pos + 1) % n
            child[insert_pos] = gene
            present[gene] = True

    def mutate(self, solution: Solution) -> None:
        blocks = self.extract_factory_blocks(solution.permutation, solution.factory_assignment)
        changed = False

        if self.rng.random() < self.sequence_mutation_rate:
            eligible_factories = [f for f, block in enumerate(blocks) if len(block) >= 2]
            if eligible_factories:
                factory = int(self.rng.choice(eligible_factories))
                seq = blocks[factory]
                same_cluster_pairs: list[tuple[int, int]] = []
                for i in range(len(seq)):
                    for j in range(i + 1, len(seq)):
                        if self.job_cluster_labels[seq[i]] == self.job_cluster_labels[seq[j]]:
                            same_cluster_pairs.append((i, j))
                if same_cluster_pairs and self.rng.random() < 0.65:
                    i, j = same_cluster_pairs[int(self.rng.integers(0, len(same_cluster_pairs)))]
                else:
                    i, j = sorted(self.rng.choice(len(seq), size=2, replace=False).tolist())
                if self.rng.random() < 0.5:
                    seq[i], seq[j] = seq[j], seq[i]
                else:
                    gene = seq.pop(j)
                    seq.insert(i, gene)
                changed = True

        if self.instance.factories > 1 and self.rng.random() < self.factory_mutation_rate:
            moved = self.cluster_guided_factory_mutation(solution, blocks)
            changed = changed or moved

        if changed:
            solution.permutation = self.flatten_factory_blocks(blocks)
        else:
            solution.permutation = self.canonicalize_permutation(solution.permutation, solution.factory_assignment)

        self.invalidate_solution(solution)

    def build_factory_loads_and_cluster_counts(
        self,
        factory_assignment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        loads = np.zeros(self.instance.factories, dtype=np.float64)
        cluster_counts = np.zeros((self.job_cluster_count, self.instance.factories), dtype=np.int64)
        for job in range(self.instance.jobs):
            factory = int(factory_assignment[job])
            cluster_id = int(self.job_cluster_labels[job])
            loads[factory] += self.mean_job_factory_processing[job, factory]
            cluster_counts[cluster_id, factory] += 1
        return loads, cluster_counts

    def compute_job_move_risk(
        self,
        factory_assignment: np.ndarray,
        loads: np.ndarray,
        cluster_counts: np.ndarray,
    ) -> np.ndarray:
        risks = np.zeros(self.instance.jobs, dtype=np.float64)
        mean_load = float(np.mean(loads)) + 1e-9
        for job in range(self.instance.jobs):
            current_factory = int(factory_assignment[job])
            cluster_id = int(self.job_cluster_labels[job])
            mismatch = self.best_job_factory_processing[job, current_factory] - np.min(self.best_job_factory_processing[job])
            rank_penalty = self.cluster_factory_rank_penalties[cluster_id, current_factory]
            load_pressure = loads[current_factory] / mean_load
            cluster_pressure = cluster_counts[cluster_id, current_factory] / max(1, self.job_cluster_sizes[cluster_id])
            urgency = self.job_urgency_score[job]
            risks[job] = (
                mismatch
                + 0.25 * self.avg_job_work[job] * rank_penalty
                + 0.25 * self.avg_job_work[job] * cluster_pressure
                + 0.20 * self.avg_job_work[job] * load_pressure * (0.35 + urgency)
            )
        return risks

    def estimate_factory_assignment_cost(
        self,
        job: int,
        factory: int,
        loads: np.ndarray,
        cluster_counts: np.ndarray,
    ) -> float:
        cluster_id = int(self.job_cluster_labels[job])
        urgency = self.job_urgency_score[job]
        scale = max(1.0, self.avg_job_work[job])
        return float(
            loads[factory]
            + 0.80 * self.mean_job_factory_processing[job, factory]
            + (0.25 + 0.20 * urgency) * self.best_job_factory_processing[job, factory]
            + self.cluster_factory_rank_penalties[cluster_id, factory] * (0.25 * scale)
            + cluster_counts[cluster_id, factory] * (0.10 * scale)
        )

    def find_cluster_aware_insert_position(self, job: int, target_block: list[int]) -> int:
        if not target_block:
            return 0
        cluster_id = int(self.job_cluster_labels[job])
        due = float(self.instance.due_dates[job])

        same_cluster_positions = [idx for idx, other in enumerate(target_block) if self.job_cluster_labels[other] == cluster_id]
        if same_cluster_positions:
            for pos in same_cluster_positions:
                if due < float(self.instance.due_dates[target_block[pos]]):
                    return pos
            return same_cluster_positions[-1] + 1

        for pos, other in enumerate(target_block):
            if due < float(self.instance.due_dates[other]):
                return pos
        return len(target_block)

    def cluster_guided_factory_mutation(self, solution: Solution, blocks: list[list[int]]) -> bool:
        loads, cluster_counts = self.build_factory_loads_and_cluster_counts(solution.factory_assignment)
        risks = self.compute_job_move_risk(solution.factory_assignment, loads, cluster_counts)
        candidate_order = np.argsort(-risks)
        mutation_points = max(1, self.instance.jobs // 10)
        moved = 0

        for job in candidate_order.tolist():
            if moved >= mutation_points:
                break
            old_factory = int(solution.factory_assignment[job])
            cluster_id = int(self.job_cluster_labels[job])

            current_cost = self.estimate_factory_assignment_cost(job, old_factory, loads, cluster_counts)
            best_factory = old_factory
            best_cost = current_cost

            preferred_factories = self.cluster_factory_preferences[cluster_id].tolist()
            candidate_factories = preferred_factories[: min(3, len(preferred_factories))]
            for factory in range(self.instance.factories):
                if factory not in candidate_factories:
                    candidate_factories.append(factory)

            for new_factory in candidate_factories:
                if new_factory == old_factory:
                    continue
                tentative_loads = loads.copy()
                tentative_counts = cluster_counts.copy()
                tentative_loads[old_factory] -= self.mean_job_factory_processing[job, old_factory]
                tentative_counts[cluster_id, old_factory] -= 1
                tentative_loads[new_factory] += self.mean_job_factory_processing[job, new_factory]
                tentative_counts[cluster_id, new_factory] += 1
                new_cost = self.estimate_factory_assignment_cost(job, new_factory, tentative_loads, tentative_counts)
                if new_cost + 1e-9 < best_cost:
                    best_cost = new_cost
                    best_factory = int(new_factory)

            improvement_threshold = 0.02 * max(1.0, self.avg_job_work[job])
            if best_factory != old_factory and best_cost + improvement_threshold < current_cost:
                old_block = blocks[old_factory]
                old_block.remove(int(job))
                insert_pos = self.find_cluster_aware_insert_position(job, blocks[best_factory])
                blocks[best_factory].insert(insert_pos, int(job))
                solution.factory_assignment[job] = best_factory
                loads[old_factory] -= self.mean_job_factory_processing[job, old_factory]
                cluster_counts[cluster_id, old_factory] -= 1
                loads[best_factory] += self.mean_job_factory_processing[job, best_factory]
                cluster_counts[cluster_id, best_factory] += 1
                moved += 1

        return moved > 0

    def tournament_selection(self, population: list[Solution]) -> Solution:
        i, j = self.rng.choice(len(population), size=2, replace=False)
        a = population[int(i)]
        b = population[int(j)]
        return a if crowded_better(a, b) else b

    def fast_non_dominated_sort(self, population: list[Solution]) -> list[list[Solution]]:
        fronts: list[list[Solution]] = []
        domination_counts = [0] * len(population)
        dominated_sets: list[list[int]] = [[] for _ in population]
        first_front: list[int] = []

        for i, p in enumerate(population):
            dominated_sets[i] = []
            domination_counts[i] = 0
            for j, q in enumerate(population):
                if i == j:
                    continue
                if dominates(p, q):
                    dominated_sets[i].append(j)
                elif dominates(q, p):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                p.rank = 0
                first_front.append(i)

        current = first_front
        rank = 0
        while current:
            fronts.append([population[idx] for idx in current])
            next_front: list[int] = []
            for p_idx in current:
                for q_idx in dominated_sets[p_idx]:
                    domination_counts[q_idx] -= 1
                    if domination_counts[q_idx] == 0:
                        population[q_idx].rank = rank + 1
                        next_front.append(q_idx)
            rank += 1
            current = next_front

        return fronts

    def assign_crowding_distance(self, fronts: list[list[Solution]]) -> None:
        for front in fronts:
            if not front:
                continue

            for solution in front:
                solution.crowding_distance = 0.0

            if len(front) <= 2:
                for solution in front:
                    solution.crowding_distance = float("inf")
                continue

            for m in range(2):
                front.sort(key=lambda sol: sol.objectives[m])
                front[0].crowding_distance = float("inf")
                front[-1].crowding_distance = float("inf")

                min_obj = front[0].objectives[m]
                max_obj = front[-1].objectives[m]
                if max_obj == min_obj:
                    continue

                for i in range(1, len(front) - 1):
                    if np.isinf(front[i].crowding_distance):
                        continue
                    prev_obj = front[i - 1].objectives[m]
                    next_obj = front[i + 1].objectives[m]
                    front[i].crowding_distance += (next_obj - prev_obj) / (max_obj - min_obj)

    def build_solution_feature_matrix(self, front: list[Solution]) -> np.ndarray:
        features: list[np.ndarray] = []
        cluster_sizes_safe = np.maximum(self.job_cluster_sizes.astype(np.float64), 1.0)

        for sol in front:
            assert sol.objectives is not None
            loads = np.zeros(self.instance.factories, dtype=np.float64)
            cluster_factory = np.zeros((self.job_cluster_count, self.instance.factories), dtype=np.float64)
            for job, factory in enumerate(sol.factory_assignment.tolist()):
                factory = int(factory)
                cluster_id = int(self.job_cluster_labels[job])
                loads[factory] += self.mean_job_factory_processing[job, factory]
                cluster_factory[cluster_id, factory] += 1.0

            load_share = loads / np.maximum(np.sum(loads), 1e-9)
            cluster_factory = cluster_factory / cluster_sizes_safe[:, None]
            tardy_ratio = 0.0
            if sol.completion_times is not None:
                tardy_ratio = float(np.mean(sol.completion_times > self.instance.due_dates))

            feat = np.concatenate(
                [
                    np.array([sol.objectives[0], sol.objectives[1], tardy_ratio], dtype=np.float64),
                    load_share.astype(np.float64),
                    cluster_factory.reshape(-1).astype(np.float64),
                ]
            )
            features.append(feat)

        matrix = np.vstack(features)
        return _safe_standardize(matrix)

    def select_by_solution_clustering(self, front: list[Solution], remaining: int) -> list[Solution]:
        if remaining <= 0:
            return []
        if len(front) <= remaining:
            return [sol.clone() for sol in front]
        if remaining == 1:
            best = max(front, key=lambda sol: (sol.crowding_distance, -sol.objectives[0], -sol.objectives[1]))
            return [best.clone()]

        feature_matrix = self.build_solution_feature_matrix(front)
        if self.solution_cluster_count > 0:
            k = min(remaining, max(2, min(self.solution_cluster_count, len(front) - 1)))
            km_result = run_kmeans(feature_matrix, k, self.rng, n_init=4, max_iter=40)
            labels = km_result.labels
        else:
            adaptive_upper = self._auto_solution_cluster_upper_bound(len(front), remaining)
            adaptive = select_adaptive_kmeans(
                feature_matrix,
                self.rng,
                min_k=2,
                max_k=adaptive_upper,
                n_init=3,
                max_iter=35,
                prefer_smaller_k_margin=0.015,
            )
            labels = adaptive.best_result.labels

        groups: dict[int, list[Solution]] = {}
        for idx, sol in enumerate(front):
            groups.setdefault(int(labels[idx]), []).append(sol)

        for solutions in groups.values():
            solutions.sort(key=lambda sol: (sol.crowding_distance, -sol.objectives[0], -sol.objectives[1]), reverse=True)

        cluster_order = sorted(
            groups.keys(),
            key=lambda cid: max(sol.crowding_distance for sol in groups[cid]),
            reverse=True,
        )

        selected: list[Solution] = []
        while len(selected) < remaining:
            progressed = False
            for cid in cluster_order:
                if groups[cid]:
                    selected.append(groups[cid].pop(0).clone())
                    progressed = True
                    if len(selected) >= remaining:
                        break
            if not progressed:
                break

        if len(selected) < remaining:
            leftovers = sorted(front, key=lambda sol: sol.crowding_distance, reverse=True)
            seen = {(tuple(sol.permutation.tolist()), tuple(sol.factory_assignment.tolist())) for sol in selected}
            for sol in leftovers:
                signature = (tuple(sol.permutation.tolist()), tuple(sol.factory_assignment.tolist()))
                if signature in seen:
                    continue
                selected.append(sol.clone())
                seen.add(signature)
                if len(selected) >= remaining:
                    break

        return selected[:remaining]

    def environmental_selection(self, fronts: list[list[Solution]], size: int) -> list[Solution]:
        new_population: list[Solution] = []
        for front in fronts:
            if len(new_population) + len(front) <= size:
                new_population.extend(sol.clone() for sol in front)
            else:
                remaining = size - len(new_population)
                new_population.extend(self.select_by_solution_clustering(front, remaining))
                break
        return new_population

    def snapshot(self, generation: int, population: list[Solution], fronts: list[list[Solution]]) -> dict[str, float]:
        best_ttd = min(sol.objectives[0] for sol in population)
        best_tec = min(sol.objectives[1] for sol in population)
        return {
            "generation": float(generation),
            "nefs": float(self.nefs),
            "population_size": float(len(population)),
            "front0_size": float(len(fronts[0]) if fronts else 0),
            "best_ttd": float(best_ttd),
            "best_tec": float(best_tec),
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def derive_output_dir(instance_path: Path, results_root: Path = Path("results")) -> Path:
    normalized = instance_path.expanduser()
    parts = list(normalized.parts)

    if "dataset" in parts:
        dataset_index = parts.index("dataset")
        relative_path = Path(*parts[dataset_index + 1 :])
    else:
        relative_path = Path(normalized.stem)

    if relative_path.suffix:
        relative_path = relative_path.with_suffix("")
    return results_root / relative_path


def deduplicate_pareto(front: list[Solution], ndigits: int = 10) -> list[Solution]:
    unique: dict[tuple[float, float], Solution] = {}
    for sol in front:
        key = (round(sol.objectives[0], ndigits), round(sol.objectives[1], ndigits))
        if key not in unique:
            unique[key] = sol
    return [unique[key] for key in sorted(unique)]


def _format_number(value: float) -> str:
    return f"{value:.6f}"


def save_results(
    output_dir: Path,
    instance: DHFSPInstance,
    pareto_front: list[Solution],
    history: list[dict[str, float]],
    solver: NSGA2DHFSPRouteC,
    config: RunConfig,
    runtime_seconds: float | None = None,
) -> None:
    ensure_dir(output_dir)

    dedup_front = deduplicate_pareto(pareto_front)

    front_csv = output_dir / "pareto_front.csv"
    with front_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "solution_id",
                "TTD",
                "TEC",
                "machine_on_time",
                "processing_time_total",
                "factory_assignment",
                "permutation",
            ]
        )
        for idx, sol in enumerate(dedup_front, start=1):
            writer.writerow(
                [
                    idx,
                    _format_number(sol.objectives[0]),
                    _format_number(sol.objectives[1]),
                    _format_number(sol.machine_on_time or 0.0),
                    _format_number(sol.processing_time_total or 0.0),
                    " ".join(str(int(x) + 1) for x in sol.factory_assignment.tolist()),
                    " ".join(str(int(x) + 1) for x in sol.permutation.tolist()),
                ]
            )

    objectives_csv = output_dir / "pareto_objectives.csv"
    with objectives_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution_id", "TTD", "TEC"])
        for idx, sol in enumerate(dedup_front, start=1):
            writer.writerow([idx, _format_number(sol.objectives[0]), _format_number(sol.objectives[1])])

    history_csv = output_dir / "history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["generation", "nefs", "population_size", "front0_size", "best_ttd", "best_tec"],
        )
        writer.writeheader()
        writer.writerows(history)

    cluster_summary = {
        str(cluster_id + 1): {
            "size": int(solver.job_cluster_sizes[cluster_id]),
            "preferred_factories": [int(x) + 1 for x in solver.cluster_factory_preferences[cluster_id].tolist()],
            "profile": solver.cluster_profiles[cluster_id],
        }
        for cluster_id in range(solver.job_cluster_count)
    }

    summary = {
        "instance_name": instance.name,
        "instance_path": str(instance.source_path) if instance.source_path else None,
        "jobs": instance.jobs,
        "stages": instance.stages,
        "factories": instance.factories,
        "machines_per_stage": instance.machines_per_stage,
        "population_size": solver.population_size,
        "max_generations": solver.max_generations,
        "seed": config.seed,
        "crossover_rate": config.crossover_rate,
        "sequence_mutation_rate": config.sequence_mutation_rate,
        "factory_mutation_rate": config.factory_mutation_rate,
        "workers": config.workers,
        "job_clusters": solver.job_cluster_count,
        "solution_clusters": solver.solution_cluster_count,
        "solution_cluster_mode": "fixed" if solver.solution_cluster_count > 0 else "adaptive_silhouette",
        "energy_saving_enabled": solver.energy_saving_enabled,
        "energy_saving_front_limit": solver.energy_saving_front_limit,
        "energy_saving_passes": solver.energy_saving_passes,
        "energy_saving_start_ratio": solver.energy_saving_start_ratio,
        "energy_saving_interval": solver.energy_saving_interval,
        "energy_saving_trials_mode": solver.energy_saving_trials_mode,
        "energy_saving_max_trials_per_solution": solver.energy_saving_max_trials_per_solution,
        "energy_saving_max_trials_per_call": solver.energy_saving_max_trials_per_call,
        "job_cluster_selection_method": solver.job_cluster_selection.method,
        "job_cluster_selection_score": solver.job_cluster_selection.best_score,
        "job_cluster_score_trace": solver.job_cluster_selection.score_trace,
        "max_generations": solver.max_generations,
        "used_nefs": solver.nefs,
        "pareto_count_raw": len(pareto_front),
        "pareto_count_unique": len(dedup_front),
        "best_ttd": min(sol.objectives[0] for sol in dedup_front),
        "best_tec": min(sol.objectives[1] for sol in dedup_front),
        "model_name": "Route-C Cluster-Driven NSGA-II for DHFSP",
        "components": [
            "job clustering for structured initialization",
            "cluster-guided factory reassignment mutation",
            "solution-space clustering for environmental selection",
            "delayed-trigger periodic energy-saving refinement on elite front solutions",
            "single-call capped forward/backward energy-saving local search",
        ],
        "clustering_model": "custom numpy K-means with k-means++ initialization and silhouette-based adaptive K selection",
        "energy_formula": "TEC = sum(machine_on_time) + 4 * sum(actual_selected_processing_time)",
        "machine_on_time_definition": (
            "For each used machine, from the first assigned operation start to the last assigned operation finish."
        ),
        "decoder_rule": (
            "At each stage, the machine is chosen by earliest completion time under machine-dependent processing "
            "times p(i,f,k,m)."
        ),
        "processing_time_semantics": "p(i,f,k,m) = workload(i) / efficiency(f,k,m)",
        "encoding_semantics": (
            "Canonical-block encoding: permutation is the concatenation of per-factory job subsequences in fixed "
            "factory order 0..F-1; only within-factory relative order carries sequencing information."
        ),
        "sequence_operator": "Factory-aware local swap/insert, preferring same-cluster jobs when possible.",
        "crossover_operator": (
            "Global OX on permutation plus uniform factory-assignment crossover, then canonical projection to block "
            "encoding."
        ),
        "energy_saving_strategy": (
            "Delayed-trigger periodic refinement: after the configured generation ratio is reached, apply "
            "energy-saving every fixed number of generations and once at the end. Each call is capped both by the "
            "number of elite solutions refined and by explicit trial limits per solution and per call."
        ),
        "cluster_summary": cluster_summary,
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds),
        "files": {
            "pareto_front_csv": str(front_csv),
            "pareto_objectives_csv": str(objectives_csv),
            "history_csv": str(history_csv),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Route-C NSGA-II for distributed hybrid flow shop scheduling with machine-heterogeneous processing times, "
            "minimizing TTD and TEC."
        )
    )
    parser.add_argument("--instance", type=Path, required=True, help="Path to a dataset TXT instance file.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory for result saving. The final folder is auto-derived from the instance path.",
    )
    parser.add_argument("--population", type=int, default=100, help="Population size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--crossover-rate", type=float, default=0.9, help="Crossover probability.")
    parser.add_argument("--sequence-mutation-rate", type=float, default=0.2, help="Permutation mutation probability.")
    parser.add_argument("--factory-mutation-rate", type=float, default=0.1, help="Factory-assignment mutation probability.")
    parser.add_argument("--job-clusters", type=int, default=0, help="Number of job clusters. Use 0 for auto.")
    parser.add_argument("--solution-clusters", type=int, default=0, help="Number of solution clusters. Use 0 for auto.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel evaluation. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--disable-energy-saving",
        action="store_true",
        help="Disable the elite full-active forward/backward energy-saving refinement.",
    )
    parser.add_argument(
        "--energy-saving-front-limit",
        type=int,
        default=10,
        help="Maximum number of current Pareto-front solutions refined in one energy-saving call. Use <=0 for all.",
    )
    parser.add_argument(
        "--energy-saving-passes",
        type=int,
        default=1,
        help="Number of forward+backward energy-saving passes applied to each refined solution.",
    )
    parser.add_argument(
        "--energy-saving-start-ratio",
        type=float,
        default=0.5,
        help="Start applying energy-saving only after completed generations / total generations reaches this ratio.",
    )
    parser.add_argument(
        "--energy-saving-interval",
        type=int,
        default=5,
        help="Apply energy-saving every N generations after the delayed trigger is active.",
    )
    return parser


def main(
    *,
    instance_path: str | Path,
    population_size: int = 100,
    max_generations: int = 400,
    crossover_rate: float = 0.9,
    sequence_mutation_rate: float = 0.2,
    factory_mutation_rate: float = 0.1,
    seed: int = 42,
    results_root: str | Path = "results",
    output_dir_override: str | Path | None = None,
    workers: int = 1,
    job_clusters: int = 0,
    solution_clusters: int = 0,
    energy_saving_enabled: bool = True,
    energy_saving_front_limit: int = 10,
    energy_saving_passes: int = 1,
    energy_saving_start_ratio: float = 0.5,
    energy_saving_interval: int = 5,
    energy_saving_max_trials_per_solution: int | None = None,
    energy_saving_max_trials_per_call: int | None = None,
) -> dict[str, object]:
    instance_path = Path(instance_path)
    results_root = Path(results_root)

    instance = parse_instance(instance_path)
    output_dir = (
        Path(output_dir_override)
        if output_dir_override is not None
        else derive_output_dir(instance_path, results_root=results_root)
    )
    population_size = max(4, int(population_size))
    max_generations = max(1, int(max_generations))

    config = RunConfig(
        instance_path=instance_path,
        output_dir=output_dir,
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
        crossover_rate=crossover_rate,
        sequence_mutation_rate=sequence_mutation_rate,
        factory_mutation_rate=factory_mutation_rate,
        workers=max(1, min(int(workers), os.cpu_count() or 1)),
        job_clusters=max(0, int(job_clusters)),
        solution_clusters=max(0, int(solution_clusters)),
        energy_saving_enabled=bool(energy_saving_enabled),
        energy_saving_front_limit=int(energy_saving_front_limit),
        energy_saving_passes=max(1, int(energy_saving_passes)),
        energy_saving_start_ratio=min(1.0, max(0.0, float(energy_saving_start_ratio))),
        energy_saving_interval=max(1, int(energy_saving_interval)),
    )

    solver = NSGA2DHFSPRouteC(
        instance=instance,
        population_size=config.population_size,
        seed=config.seed,
        max_generations=config.max_generations,
        crossover_rate=config.crossover_rate,
        sequence_mutation_rate=config.sequence_mutation_rate,
        factory_mutation_rate=config.factory_mutation_rate,
        workers=config.workers,
        job_clusters=config.job_clusters,
        solution_clusters=config.solution_clusters,
        energy_saving_enabled=config.energy_saving_enabled,
        energy_saving_front_limit=config.energy_saving_front_limit,
        energy_saving_passes=config.energy_saving_passes,
        energy_saving_start_ratio=config.energy_saving_start_ratio,
        energy_saving_interval=config.energy_saving_interval,
    )

    started_at = time.time()
    pareto_front, history = solver.solve()
    runtime_seconds = time.time() - started_at
    save_results(config.output_dir, instance, pareto_front, history, solver, config, runtime_seconds=runtime_seconds)

    unique_front = deduplicate_pareto(pareto_front)
    best_ttd = float(min(sol.objectives[0] for sol in unique_front))
    best_tec = float(min(sol.objectives[1] for sol in unique_front))

    print(f"Instance name: {instance.name}")
    print(f"Instance path: {instance_path}")
    print(f"Jobs={instance.jobs}, Stages={instance.stages}, Factories={instance.factories}")
    print(f"Population size: {config.population_size}")
    print(f"Max generations: {solver.max_generations}")
    print(f"Crossover rate: {config.crossover_rate}")
    print(f"Sequence mutation rate: {config.sequence_mutation_rate}")
    print(f"Factory mutation rate: {config.factory_mutation_rate}")
    print(f"Job clusters: {solver.job_cluster_count}")
    if solver.solution_cluster_count > 0:
        print(f"Solution clusters: {solver.solution_cluster_count}")
    else:
        print("Solution clusters: adaptive silhouette mode")
    print(f"Seed: {config.seed}")
    print(f"Workers: {config.workers}")
    print(f"Energy saving enabled: {solver.energy_saving_enabled}")
    print(f"Energy saving front limit: {solver.energy_saving_front_limit}")
    print(f"Energy saving passes: {solver.energy_saving_passes}")
    print(f"Energy saving start ratio: {solver.energy_saving_start_ratio}")
    print(f"Energy saving interval: {solver.energy_saving_interval}")
    print(f"Energy saving trials mode: {solver.energy_saving_trials_mode}")
    print(f"Energy saving max trials per solution: {solver.energy_saving_max_trials_per_solution}")
    print(f"Energy saving max trials per call: {solver.energy_saving_max_trials_per_call}")
    print(f"Stop criterion: Fixed generations = {solver.max_generations}")
    print(f"Used NEFs: {solver.nefs}")
    print(f"Unique Pareto solutions: {len(unique_front)}")
    print(f"Best TTD: {_format_number(best_ttd)}")
    print(f"Best TEC: {_format_number(best_tec)}")
    print(f"Results saved to: {config.output_dir}")

    return {
        "instance_name": instance.name,
        "instance_path": str(instance_path),
        "output_dir": str(config.output_dir),
        "population_size": config.population_size,
        "max_generations": solver.max_generations,
        "crossover_rate": config.crossover_rate,
        "sequence_mutation_rate": config.sequence_mutation_rate,
        "factory_mutation_rate": config.factory_mutation_rate,
        "job_clusters": solver.job_cluster_count,
        "solution_clusters": solver.solution_cluster_count,
        "solution_cluster_mode": "fixed" if solver.solution_cluster_count > 0 else "adaptive_silhouette",
        "job_cluster_selection_method": solver.job_cluster_selection.method,
        "job_cluster_selection_score": solver.job_cluster_selection.best_score,
        "job_cluster_score_trace": solver.job_cluster_selection.score_trace,
        "seed": config.seed,
        "workers": config.workers,
        "energy_saving_enabled": solver.energy_saving_enabled,
        "energy_saving_front_limit": solver.energy_saving_front_limit,
        "energy_saving_passes": solver.energy_saving_passes,
        "energy_saving_start_ratio": solver.energy_saving_start_ratio,
        "energy_saving_interval": solver.energy_saving_interval,
        "energy_saving_trials_mode": solver.energy_saving_trials_mode,
        "energy_saving_max_trials_per_solution": solver.energy_saving_max_trials_per_solution,
        "energy_saving_max_trials_per_call": solver.energy_saving_max_trials_per_call,
        "runtime_seconds": float(runtime_seconds),
        "max_generations": solver.max_generations,
        "used_nefs": solver.nefs,
        "pareto_count_unique": len(unique_front),
        "best_ttd": best_ttd,
        "best_tec": best_tec,
    }


NSGA2DHFSP = NSGA2DHFSPRouteC


def _read_index_file(index_path: Path) -> list[Path]:
    entries: list[Path] = []
    for raw_line in index_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append((index_path.parent / line).resolve())
    return entries


def discover_batch_instances(dataset_root: Path) -> dict[str, list[Path]]:
    dataset_root = dataset_root.resolve()
    groups = {
        "small": dataset_root / "comparison_experiment" / "small",
        "large": dataset_root / "comparison_experiment" / "large",
        "realcase": dataset_root / "realcase",
    }
    discovered: dict[str, list[Path]] = {}
    for scale, group_dir in groups.items():
        index_path = group_dir / "index.txt"
        if index_path.exists():
            paths = _read_index_file(index_path)
        else:
            paths = sorted(group_dir.glob("*.txt"))
        valid_paths = [path for path in paths if path.name.lower() != "index.txt"]
        if not valid_paths:
            raise FileNotFoundError(f"No dataset TXT files found for {scale}: {group_dir}")
        discovered[scale] = valid_paths
    return discovered


def extract_dataset_case_id(instance_path: Path) -> str:
    match = re.search(r"(\d+)(?!.*\d)", instance_path.stem)
    if match is None:
        raise ValueError(f"Cannot extract numeric dataset id from: {instance_path.name}")
    return match.group(1)


def build_batch_output_dir(results_root: Path, scale: str, instance_path: Path, seed: int) -> Path:
    case_id = extract_dataset_case_id(instance_path)
    return results_root / scale / case_id / f"seed_{seed}"


def write_batch_summary(results_root: Path, batch_summary: list[dict[str, object]], batch_manifest: dict[str, object]) -> None:
    ensure_dir(results_root)

    summary_csv = results_root / "batch_summary.csv"
    fieldnames = [
        "status",
        "scale",
        "case_id",
        "seed",
        "instance_name",
        "instance_path",
        "output_dir",
        "runtime_seconds",
        "pareto_count_unique",
        "best_ttd",
        "best_tec",
        "error",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in batch_summary:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    manifest_path = results_root / "batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class F5RunConfig:
    instance_path: str = ""
    results_root: str = "result"
    population_size: int = 100
    max_generations: int = 400
    seed: int = 0
    seed_start: int = 0
    seed_end: int = 19
    crossover_rate: float = 0.9
    sequence_mutation_rate: float = 0.2
    factory_mutation_rate: float = 0.1
    workers: int = 1
    job_clusters: int = 0
    solution_clusters: int = 0
    energy_saving_enabled: bool = True
    energy_saving_front_limit: int = 10
    energy_saving_passes: int = 1
    energy_saving_start_ratio: float = 0.5
    energy_saving_interval: int = 5
    batch_mode: bool = True
    skip_completed: bool = True
    auto_pick_instance_when_empty: bool = False
    show_finish_message: bool = True


def _select_instance_path_interactive() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            title="Select a DHFSP instance TXT file",
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")],
        )
        root.destroy()
        return selected or ""
    except Exception:
        return ""


def run_f5_batch(config: F5RunConfig) -> dict[str, object]:
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / "dataset"
    results_root = (script_dir / config.results_root).resolve()
    discovered = discover_batch_instances(dataset_root)
    seeds = list(range(int(config.seed_start), int(config.seed_end) + 1))
    total_runs = sum(len(paths) for paths in discovered.values()) * len(seeds)

    batch_summary: list[dict[str, object]] = []
    completed_runs = 0
    skipped_runs = 0
    failed_runs = 0

    print(f"Batch mode enabled. Total planned runs: {total_runs}")
    for scale in ("small", "large", "realcase"):
        print(f"  - {scale}: {len(discovered[scale])} datasets")

    run_index = 0
    for scale in ("small", "large", "realcase"):
        for instance_path in discovered[scale]:
            case_id = extract_dataset_case_id(instance_path)
            for seed in seeds:
                run_index += 1
                output_dir = build_batch_output_dir(results_root, scale, instance_path, seed)
                summary_path = output_dir / "summary.json"
                if config.skip_completed and summary_path.exists():
                    skipped_runs += 1
                    print(
                        f"[{run_index}/{total_runs}] Skip existing run: scale={scale}, case={case_id}, seed={seed}, "
                        f"dir={output_dir}"
                    )
                    try:
                        cached = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        cached = {}
                    batch_summary.append(
                        {
                            "status": "skipped",
                            "scale": scale,
                            "case_id": case_id,
                            "seed": seed,
                            "instance_name": cached.get("instance_name", instance_path.stem),
                            "instance_path": str(instance_path),
                            "output_dir": str(output_dir),
                            "runtime_seconds": cached.get("runtime_seconds", ""),
                            "pareto_count_unique": cached.get("pareto_count_unique", ""),
                            "best_ttd": cached.get("best_ttd", ""),
                            "best_tec": cached.get("best_tec", ""),
                            "error": "",
                        }
                    )
                    continue

                print(
                    f"[{run_index}/{total_runs}] Running scale={scale}, case={case_id}, seed={seed} -> {output_dir}"
                )
                try:
                    result = main(
                        instance_path=instance_path,
                        population_size=config.population_size,
                        max_generations=config.max_generations,
                        crossover_rate=config.crossover_rate,
                        sequence_mutation_rate=config.sequence_mutation_rate,
                        factory_mutation_rate=config.factory_mutation_rate,
                        seed=seed,
                        results_root=results_root,
                        output_dir_override=output_dir,
                        workers=config.workers,
                        job_clusters=config.job_clusters,
                        solution_clusters=config.solution_clusters,
                        energy_saving_enabled=config.energy_saving_enabled,
                        energy_saving_front_limit=config.energy_saving_front_limit,
                        energy_saving_passes=config.energy_saving_passes,
                        energy_saving_start_ratio=config.energy_saving_start_ratio,
                        energy_saving_interval=config.energy_saving_interval,
                    )
                    completed_runs += 1
                    batch_summary.append(
                        {
                            "status": "completed",
                            "scale": scale,
                            "case_id": case_id,
                            "seed": seed,
                            "instance_name": result.get("instance_name", instance_path.stem),
                            "instance_path": str(instance_path),
                            "output_dir": result.get("output_dir", str(output_dir)),
                            "runtime_seconds": result.get("runtime_seconds", ""),
                            "pareto_count_unique": result.get("pareto_count_unique", ""),
                            "best_ttd": result.get("best_ttd", ""),
                            "best_tec": result.get("best_tec", ""),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    failed_runs += 1
                    ensure_dir(output_dir)
                    error_payload = {
                        "status": "failed",
                        "scale": scale,
                        "case_id": case_id,
                        "seed": seed,
                        "instance_path": str(instance_path),
                        "output_dir": str(output_dir),
                        "error": str(exc),
                    }
                    (output_dir / "error.json").write_text(
                        json.dumps(error_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    batch_summary.append(
                        {
                            "status": "failed",
                            "scale": scale,
                            "case_id": case_id,
                            "seed": seed,
                            "instance_name": instance_path.stem,
                            "instance_path": str(instance_path),
                            "output_dir": str(output_dir),
                            "runtime_seconds": "",
                            "pareto_count_unique": "",
                            "best_ttd": "",
                            "best_tec": "",
                            "error": str(exc),
                        }
                    )
                    print(f"Run failed for scale={scale}, case={case_id}, seed={seed}: {exc}")

    batch_manifest = {
        "mode": "f5_batch",
        "results_root": str(results_root),
        "dataset_root": str(dataset_root),
        "seed_start": int(config.seed_start),
        "seed_end": int(config.seed_end),
        "seed_count": len(seeds),
        "planned_run_count": total_runs,
        "completed_runs": completed_runs,
        "skipped_runs": skipped_runs,
        "failed_runs": failed_runs,
        "datasets": {scale: [str(path) for path in paths] for scale, paths in discovered.items()},
        "population_size": config.population_size,
        "max_generations": config.max_generations,
        "crossover_rate": config.crossover_rate,
        "sequence_mutation_rate": config.sequence_mutation_rate,
        "factory_mutation_rate": config.factory_mutation_rate,
        "workers": config.workers,
        "job_clusters": config.job_clusters,
        "solution_clusters": config.solution_clusters,
        "energy_saving_enabled": config.energy_saving_enabled,
        "energy_saving_front_limit": config.energy_saving_front_limit,
        "energy_saving_passes": config.energy_saving_passes,
        "energy_saving_start_ratio": config.energy_saving_start_ratio,
        "energy_saving_interval": config.energy_saving_interval,
    }
    write_batch_summary(results_root, batch_summary, batch_manifest)

    result = {
        "mode": "batch",
        "results_root": str(results_root),
        "planned_run_count": total_runs,
        "completed_runs": completed_runs,
        "skipped_runs": skipped_runs,
        "failed_runs": failed_runs,
        "batch_summary_csv": str(results_root / "batch_summary.csv"),
        "batch_manifest_json": str(results_root / "batch_manifest.json"),
    }
    if failed_runs > 0:
        raise RuntimeError(
            f"Batch completed with {failed_runs} failed runs. Check {result['batch_summary_csv']} for details."
        )
    return result


def run_f5_mode(config: F5RunConfig) -> dict[str, object]:
    if config.batch_mode:
        result = run_f5_batch(config)
        if config.show_finish_message:
            try:
                import tkinter as tk
                from tkinter import messagebox

                root = tk.Tk()
                root.withdraw()
                root.update()
                messagebox.showinfo(
                    title="Route-C batch finished",
                    message=(
                        f"Planned runs: {result['planned_run_count']}\n"
                        f"Completed: {result['completed_runs']}\n"
                        f"Skipped: {result['skipped_runs']}\n"
                        f"Failed: {result['failed_runs']}\n"
                        f"Summary CSV: {result['batch_summary_csv']}"
                    ),
                )
                root.destroy()
            except Exception:
                pass
        return result

    instance_path = config.instance_path.strip()
    if not instance_path and config.auto_pick_instance_when_empty:
        instance_path = _select_instance_path_interactive()

    if not instance_path:
        raise ValueError(
            "No instance file was provided. In F5 mode, either set F5_CONFIG.instance_path at the bottom of the "
            "script or choose a file from the pop-up dialog."
        )

    result = main(
        instance_path=instance_path,
        population_size=config.population_size,
        max_generations=config.max_generations,
        crossover_rate=config.crossover_rate,
        sequence_mutation_rate=config.sequence_mutation_rate,
        factory_mutation_rate=config.factory_mutation_rate,
        seed=config.seed,
        results_root=config.results_root,
        workers=config.workers,
        job_clusters=config.job_clusters,
        solution_clusters=config.solution_clusters,
        energy_saving_enabled=config.energy_saving_enabled,
        energy_saving_front_limit=config.energy_saving_front_limit,
        energy_saving_passes=config.energy_saving_passes,
        energy_saving_start_ratio=config.energy_saving_start_ratio,
        energy_saving_interval=config.energy_saving_interval,
    )

    if config.show_finish_message:
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.update()
            messagebox.showinfo(
                title="Route-C finished",
                message=(
                    f"Instance: {result['instance_name']}\n"
                    f"Unique Pareto count: {result['pareto_count_unique']}\n"
                    f"Best TTD: {result['best_ttd']:.6f}\n"
                    f"Best TEC: {result['best_tec']:.6f}\n"
                    f"Results folder: {result['output_dir']}"
                ),
            )
            root.destroy()
        except Exception:
            pass

    return result


def cli() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    main(
        instance_path=args.instance,
        population_size=args.population,
        crossover_rate=args.crossover_rate,
        sequence_mutation_rate=args.sequence_mutation_rate,
        factory_mutation_rate=args.factory_mutation_rate,
        seed=args.seed,
        results_root=args.results_root,
        workers=args.workers,
        job_clusters=args.job_clusters,
        solution_clusters=args.solution_clusters,
        energy_saving_enabled=not args.disable_energy_saving,
        energy_saving_front_limit=args.energy_saving_front_limit,
        energy_saving_passes=args.energy_saving_passes,
        energy_saving_start_ratio=args.energy_saving_start_ratio,
        energy_saving_interval=args.energy_saving_interval,
    )


F5_CONFIG = F5RunConfig(
    instance_path="",
    results_root="result",
    population_size=160,
    max_generations=400,
    seed=0,
    seed_start=0,
    seed_end=19,
    crossover_rate=0.85,
    sequence_mutation_rate=0.3,
    factory_mutation_rate=0.06,
    workers=8,
    job_clusters=0,
    solution_clusters=0,
    energy_saving_enabled=True,
    energy_saving_front_limit=12,
    energy_saving_passes=1,
    energy_saving_start_ratio=0.6,
    energy_saving_interval=3,
    batch_mode=True,
    skip_completed=True,
    auto_pick_instance_when_empty=False,
    show_finish_message=True,
)


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.update()
        if error:
            messagebox.showerror(title=title, message=message)
        else:
            messagebox.showinfo(title=title, message=message)
        root.destroy()
    except Exception:
        print(f"[{title}] {message}")


if __name__ == "__main__":
    start_time = time.time()
    mp.freeze_support()
    try:
        if len(sys.argv) > 1:
            cli()
        else:
            run_f5_mode(F5_CONFIG)
    except Exception as exc:
        _show_message("Route-C run failed", str(exc), error=True)
        raise
    end=time.time()
    elapsed = end - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")
