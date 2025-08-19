#!/usr/bin/env python3
import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Union,Literal


import numpy as np
import torch
import copy
import random

# repo bootstrap
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.graph.graph import Graph
from AgentDropout.tools.reader.readers import JSONReader
from AgentDropout.utils.globals import Time
from experiments.accuracy import Accuracy
from datasets.gsm8k_dataset import multiarith_data_process


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# EA config
# =========================
@dataclass
class MaskEAArgs:
    population_size: int
    evo_iter: int
    parent_size: int
    mutation_size: int
    crossover_size: int
    mutation_prob: float
    target_sparsity: float
    eval_batch: int
    num_rounds: int
    seed: int


# =========================
# Converter: gene <-> masks
# =========================
class Converter:
    """
    Encodes masks as a flat binary genome and decodes back to:
      {'spatial': Tensor or List[Tensor], 'temporal': Tensor or List[Tensor]}.
    """
    def __init__(self, spatial_shapes: List[tuple], temporal_shapes: List[tuple]):
        self.s_shapes = [tuple(s) for s in spatial_shapes]
        self.t_shapes = [tuple(t) for t in temporal_shapes]
        self.s_numels = [int(np.prod(s)) for s in self.s_shapes]
        self.t_numels = [int(np.prod(t)) for t in self.t_shapes]
        self.gene_len = sum(self.s_numels) + sum(self.t_numels)

    def gene2config(self, gene: List[int]) -> Dict[str, Any]:
        ptr = 0
        spatial, temporal = [], []
        # spatial
        for n, shape in zip(self.s_numels, self.s_shapes):
            arr = torch.tensor(gene[ptr:ptr+n], dtype=torch.int8).reshape(shape)
            if arr.ndim == 2:
                arr = arr.clone(); arr.fill_diagonal_(0)  # forbid self-loops (optional)
            spatial.append(arr); ptr += n
        # temporal
        for n, shape in zip(self.t_numels, self.t_shapes):
            arr = torch.tensor(gene[ptr:ptr+n], dtype=torch.int8).reshape(shape)
            if arr.ndim == 2:
                arr = arr.clone(); arr.fill_diagonal_(0)
            temporal.append(arr); ptr += n
        if len(spatial) == 1: spatial = spatial[0]
        if len(temporal) == 1: temporal = temporal[0]
        return {"spatial": spatial, "temporal": temporal}

    def config2gene(self, cfg: Dict[str, Any]) -> List[int]:
        def to_list(t):
            if isinstance(t, list): return t
            return [t]
        bits: List[int] = []
        for t in to_list(cfg["spatial"]):
            bits.extend(t.reshape(-1).tolist())
        for t in to_list(cfg["temporal"]):
            bits.extend(t.reshape(-1).tolist())
        return [int(x) for x in bits]


# =========================
# Scoring (async eval)
# =========================

async def score_config_async(graph: Graph, dataset, cfg: Dict[str, Any],
                             eval_batch: int, num_rounds: int,
                             req_timeout: float = 45.0) -> float:
    g = copy.deepcopy(graph)

    # apply masks
    if not g.diff:
        g.spatial_masks = cfg["spatial"]
        g.temporal_masks = cfg["temporal"]
    else:
        g.spatial_masks = [m.clone() for m in cfg["spatial"]]
        g.temporal_masks = [m.clone() for m in cfg["temporal"]]

    n = min(eval_batch, len(dataset))
    tasks, golds = [], []
    # print(f"[EVO] Scoring {n} samples with {num_rounds} rounds...")  # keep logs light

    async def _one(inp):
        try:
            return await asyncio.wait_for(g.arun(inp, num_rounds), timeout=req_timeout)
        except Exception:
            return None  # treat failure/timeout as incorrect

    for i in range(n):
        rec = dataset[i]
        inp = {"task": rec["task"]}  # matches your working code
        golds.append(rec["answer"])
        tasks.append(_one(inp))

    results = await asyncio.gather(*tasks, return_exceptions=False)

    acc_sum, ok = 0.0, 0
    for res, y in zip(results, golds):
        if res is None:
            continue  # timeout/error → 0 credit
        raw_answer, _ = res
        ans = str(raw_answer)
        acc = Accuracy(); acc.update(ans, str(y))
        acc_sum += acc.get()
        ok += 1

    # if all failed (server hiccup), return a large loss so EA moves on
    if ok == 0:
        return 1e6

    avg_acc = acc_sum / ok
    return -avg_acc



# =========================
# Evolutionary algorithm
# =========================
class Evolution:
    """
    Stand-alone EA (no fairseq). Mirrors the HAT knobs and flow:
      - population = parents + mutations + crossovers
      - selection by ascending loss
      - genes are binary, decoded by Converter
    """

    def __init__(self, args: MaskEAArgs, graph: Graph, dataset, converter: Converter, sparsity_tol: float = 0.10):
        self.args = args
        self.graph = graph
        self.dataset = dataset
        self.conv = converter
        self.keep_prob = 1.0 - args.target_sparsity
        self.sparsity_tol = sparsity_tol

        assert args.population_size == args.parent_size + args.mutation_size + args.crossover_size, \
            "population_size must equal parent_size + mutation_size + crossover_size"

    # --- genome utils ---
    def _rand_gene(self) -> List[int]:
        # sample each bit ~ Bernoulli(keep_prob)
        return [1 if random.random() < self.keep_prob else 0 for _ in range(self.conv.gene_len)]

    def _satisfy_sparsity(self, gene: List[int]) -> bool:
        cfg = self.conv.gene2config(gene)
        def frac_zero(t):
            if isinstance(t, list):
                t = t[0]
            return float((t == 0).float().mean().item())
        tgt = self.args.target_sparsity
        s_ok = abs(frac_zero(cfg["spatial"]) - tgt) <= self.sparsity_tol
        t_ok = abs(frac_zero(cfg["temporal"]) - tgt) <= self.sparsity_tol
        return s_ok and t_ok

    def random_sample(self, k: int) -> List[List[int]]:
        pop = []
        while len(pop) < k:
            g = self._rand_gene()
            if self._satisfy_sparsity(g):
                pop.append(g)
        return pop

    def mutate(self, gene: List[int]) -> List[int]:
        p = self.args.mutation_prob
        return [1 - b if random.random() < p else b for b in gene]

    def crossover(self, a: List[int], b: List[int]) -> List[int]:
        return [random.choice((aa, bb)) for aa, bb in zip(a, b)]


    async def _get_scores_async(self, genes):
        cfgs = [self.conv.gene2config(g) for g in genes]
        scores = []
        # evaluate at most eval_parallel configs concurrently
        for i in range(0, len(cfgs), getattr(self.args, "eval_parallel", 4)):
            batch = cfgs[i:i + getattr(self.args, "eval_parallel", 4)]
            batch_scores = await asyncio.gather(*[
                score_config_async(
                    self.graph, self.dataset, cfg,
                    self.args.eval_batch, self.args.num_rounds,
                    req_timeout=getattr(self.args, "req_timeout", 45.0)
                )
                for cfg in batch
            ])
            scores.extend(batch_scores)
        return scores


    async def run_async(self):
        pop = self.random_sample(self.args.population_size)
        best_cfg, best_score = None, float('inf')

        for it in range(self.args.evo_iter):
            scores = await self._get_scores_async(pop)
            order = np.argsort(scores)  # lower is better
            parents = [pop[i] for i in order[:self.args.parent_size]]

            if scores[order[0]] < best_score:
                best_score = scores[order[0]]
                best_cfg = self.conv.gene2config(parents[0])

            print(f"[EVO] iter={it} best_loss={scores[order[0]]:.4f} median_loss={np.median(scores):.4f}")

            # mutations
            muts = []
            while len(muts) < self.args.mutation_size:
                cand = self.mutate(random.choice(parents))
                if self._satisfy_sparsity(cand):
                    muts.append(cand)

            # crossovers
            xovers = []
            while len(xovers) < self.args.crossover_size:
                a, b = random.sample(parents, 2)
                cand = self.crossover(a, b)
                if self._satisfy_sparsity(cand):
                    xovers.append(cand)

            pop = parents + muts + xovers

        return best_cfg


def edge_vector_shapes(graph):
    """
    Returns shapes for masks as 1-D vectors over the potential edges.
    For diff=True, one vector per round/param; for diff=False, just one vector.
    """
    s_len = len(graph.potential_spatial_edges)
    t_len = len(graph.potential_temporal_edges)

    if not graph.diff:
        s_shapes = [(s_len,)]
        t_shapes = [(t_len,)]
    else:
        s_shapes = [(s_len,)] * len(graph.spatial_logits)
        t_shapes = [(t_len,)] * len(graph.temporal_logits)
    return s_shapes, t_shapes

def determine_search_shapes(graph, num_rounds: int):
    """
    Build 1-D vector shapes over potential edges.
    For diff=True, repeat one vector per round (length R).
    """
    s_len = len(graph.potential_spatial_edges)
    t_len = len(graph.potential_temporal_edges)

    if graph.diff:
        # Prefer true R if logits are already per-round; otherwise fall back to num_rounds
        R_s = len(graph.spatial_logits) if isinstance(graph.spatial_logits, list) else num_rounds
        R_t = len(graph.temporal_logits) if isinstance(graph.temporal_logits, list) else num_rounds
        # If they differ, use the rounds arg to keep them aligned
        R = num_rounds
        s_shapes = [(s_len,)] * R
        t_shapes = [(t_len,)] * R
    else:
        s_shapes = [(s_len,)]
        t_shapes = [(t_len,)]
    return s_shapes, t_shapes

# =========================
# Main
# =========================
async def main(args):
    set_seed(args.seed)

    # ---- data ----
    dataset = JSONReader.parse_file(args.dataset_json)
    dataset = multiarith_data_process(dataset)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/MultiArith")
    result_dir.mkdir(parents=True, exist_ok=True)
    out_pt = result_dir / f"best_masks_{current_time}.pt"
    out_txt = result_dir / f"best_masks_{current_time}.txt"

    # ---- agents & graph ----
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode, len(agent_names))

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
        llm_list = list(cfg_json["model_list"].keys())

    graph = Graph(
        domain=args.domain,
        llm_list=llm_list,
        agent_names=agent_names,
        decision_method=decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        rounds=args.num_rounds,
        diff=args.diff,
        dec=args.dec,
        **kwargs
    )

    # ---- converter shapes from graph logits ----
    # if not graph.diff:
    #     s_shapes = [tuple(graph.spatial_logits.shape)]
    #     t_shapes = [tuple(graph.temporal_logits.shape)]
    # else:
    #     s_shapes = [tuple(x.shape) for x in graph.spatial_logits]
    #     t_shapes = [tuple(x.shape) for x in graph.temporal_logits]
    s_shapes, t_shapes = determine_search_shapes(graph, args.num_rounds)
    converter = Converter(s_shapes, t_shapes)

    # ---- EA config (default target_sparsity = pruning_rate) ----
    target_sparsity = args.target_sparsity if args.target_sparsity is not None else args.pruning_rate
    ea_cfg = MaskEAArgs(
        population_size=args.population_size,
        evo_iter=args.evo_iter,
        parent_size=args.parent_size,
        mutation_size=args.mutation_size,
        crossover_size=args.crossover_size,
        mutation_prob=args.mutation_prob,
        target_sparsity=target_sparsity,
        eval_batch=args.eval_batch,
        num_rounds=args.num_rounds,
        seed=args.seed,
    )

    evolver = Evolution(ea_cfg, graph, dataset, converter)
    best = await evolver.run_async()  # {'spatial': ..., 'temporal': ...}

    # ---- report & save ----
    def sparsity(t):
        if isinstance(t, list): t = t[0]
        return float((t == 0).float().mean().item())

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# AgentDrop EA result\n")
        if not graph.diff:
            f.write(f"spatial_shape: {tuple(best['spatial'].shape)}\n")
            f.write(f"temporal_shape: {tuple(best['temporal'].shape)}\n")
        else:
            f.write(f"num_spatial_parts: {len(best['spatial'])}\n")
            f.write(f"num_temporal_parts: {len(best['temporal'])}\n")
        f.write(f"spatial_sparsity: {sparsity(best['spatial']):.4f}\n")
        f.write(f"temporal_sparsity: {sparsity(best['temporal']):.4f}\n")

    torch.save(best, out_pt)
    print(f"\nSaved best masks:\n - {out_txt}\n - {out_pt}")


# =========================
# CLI
# =========================
def cli_main():
    p = argparse.ArgumentParser("AgentDrop — Evolutionary mask search (no fairseq)")
    p.add_argument("--dataset_json", type=str, default="datasets/MultiArith/test.json")
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--domain", type=str, default="gsm8k")

    p.add_argument('--mode', type=str, default='FullConnected',
                   choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'])
    p.add_argument('--decision_method', type=str, default='FinalRefer')
    p.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    p.add_argument('--agent_nums', nargs='+', type=int, default=[4])

    p.add_argument('--optimized_spatial', action='store_true')
    p.add_argument('--optimized_temporal', action='store_true')
    p.add_argument('--diff', action='store_true')
    p.add_argument('--dec', action='store_true')

    # EA knobs (HAT-like)
    p.add_argument('--evo-iter', type=int, default=30)
    p.add_argument('--population-size', type=int, default=125)
    p.add_argument('--parent-size', type=int, default=25)
    p.add_argument('--mutation-size', type=int, default=50)
    p.add_argument('--crossover-size', type=int, default=50)
    p.add_argument('--mutation-prob', type=float, default=0.3)

    # scoring & sparsity
    p.add_argument('--eval-batch', type=int, default=6)
    p.add_argument('--pruning_rate', type=float, default=0.25, help="fallback for target_sparsity")
    p.add_argument('--target-sparsity', type=float, default=None, help="fraction of zeros in masks")
    p.add_argument('--num_rounds', type=int, default=1)

    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--eval-parallel', type=int, default=4,
               help='how many configs to score concurrently')
    p.add_argument('--req-timeout', type=float, default=45.0,
               help='seconds timeout per sample request')

    args = p.parse_args()

    # sanity: population sizing
    assert args.population_size == args.parent_size + args.mutation_size + args.crossover_size, \
        "population_size must equal parent_size + mutation_size + crossover_size"

    asyncio.run(main(args))

def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        # random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Math Solver'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}  

if __name__ == "__main__":
    cli_main()
