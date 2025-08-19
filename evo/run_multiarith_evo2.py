import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List,Union,Literal, Tuple, Dict, Any
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.graph.graph import Graph
from AgentDropout.tools.reader.readers import JSONReader, JSONLReader
from AgentDropout.utils.globals import Time
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentDropout.utils.utils import nuclear_norm,frobenius_norm
from datasets.gsm8k_dataset import svamp_data_process,gsm_get_predict, gsm_data_process,multiarith_data_process
from datasets.aqua_dataset import aqua_data_process,aqua_get_predict
from AgentDropout.agents.agent_registry import AgentRegistry

# --------------------------
# Utility helpers (EA)
# --------------------------

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)
    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

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

# --------------------------
# EA Building Blocks
# --------------------------

TensorOrList = Union[torch.Tensor, List[torch.Tensor]]

# ---- simple EA helpers ----

def _clone_param(p):
    """Return a candidate-ready copy:
       - tensor -> Tensor
       - (list/tuple/ParameterList) -> list[Tensor]
    """
    if isinstance(p, (list, tuple, nn.ParameterList)):
        return [t.detach().clone() for t in list(p)]
    return p.detach().clone()

def _uniform_crossover(a, b):
    """Elementwise uniform crossover for Tensor or list[Tensor]."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        mask = torch.bernoulli(torch.full_like(a, 0.5))
        return a * mask + b * (1 - mask)
    # list case
    out = []
    for x, y in zip(a, b):
        mask = torch.bernoulli(torch.full_like(x, 0.5))
        out.append(x * mask + y * (1 - mask))
    return out

def _mutate(p, mutation_rate: float, mutation_std: float):
    """Sparse Gaussian mutation for Tensor or list[Tensor]."""
    def m(t):
        mask = torch.bernoulli(torch.full_like(t, mutation_rate))
        noise = torch.randn_like(t) * mutation_std
        return torch.clamp(t + mask * noise, -8.0, 8.0)
    if isinstance(p, torch.Tensor):
        return m(p)
    return [m(t) for t in p]

def _reshape_for_penalties(param, n: int):
    """Reshape to (n,n) for penalties; supports Tensor or list[Tensor]."""
    if isinstance(param, torch.Tensor):
        return param.reshape((n, n))
    return [t.reshape((n, n)) for t in param]


async def _evaluate_candidate(
    base_graph: Graph,
    candidate: Dict[str, TensorOrList],
    records: List[Dict[str, Any]],
    args,
    kwargs,
    agent_names: List[str],
    *,
    is_dec: bool
) -> Tuple[float, float, float]:
    """
    Returns (fitness, accuracy, penalty)
    """
    realized_graph = copy.deepcopy(base_graph)

    # Inject candidate params into realized graph
    if is_dec:
        if realized_graph.diff:
            realized_graph.spatial_logits_1  = nn.ParameterList([nn.Parameter(t) for t in candidate["spatial"]])
            realized_graph.temporal_logits_1 = nn.ParameterList([nn.Parameter(t) for t in candidate["temporal"]])
        else:
            realized_graph.spatial_logits_1  = candidate["spatial"]
            realized_graph.temporal_logits_1 = candidate["temporal"]
    else:
        if realized_graph.diff:
            realized_graph.spatial_logits  = nn.ParameterList([nn.Parameter(t) for t in candidate["spatial"]])
            realized_graph.temporal_logits = nn.ParameterList([nn.Parameter(t) for t in candidate["temporal"]])
        else:
            realized_graph.spatial_logits  = candidate["spatial"]
            realized_graph.temporal_logits = candidate["temporal"]

    # Penalty terms (structure regularization)
    with torch.no_grad():
        n = sum(args.agent_nums)
        fixed_s = torch.tensor(kwargs["fixed_spatial_masks"], dtype=torch.float32).reshape((len(agent_names), len(agent_names)))
        fixed_t = torch.tensor(kwargs["fixed_temporal_masks"], dtype=torch.float32).reshape((len(agent_names), len(agent_names)))

        if is_dec:
            Sm = _reshape_for_penalties(realized_graph.spatial_logits_1, n) if not realized_graph.diff else \
                 [p.reshape((n, n)) for p in realized_graph.spatial_logits_1]
            Tm = _reshape_for_penalties(realized_graph.temporal_logits_1, n) if not realized_graph.diff else \
                 [p.reshape((n, n)) for p in realized_graph.temporal_logits_1]
        else:
            Sm = _reshape_for_penalties(realized_graph.spatial_logits, n) if not realized_graph.diff else \
                 [p.reshape((n, n)) for p in realized_graph.spatial_logits]
            Tm = _reshape_for_penalties(realized_graph.temporal_logits, n) if not realized_graph.diff else \
                 [p.reshape((n, n)) for p in realized_graph.temporal_logits]

        if not realized_graph.diff:
            loss_s = nuclear_norm(Sm)
            loss_t = nuclear_norm(Tm)
            frob_s = frobenius_norm(fixed_s, Sm)
            frob_t = frobenius_norm(fixed_t, Tm)
        else:
            loss_s = torch.mean(torch.stack([nuclear_norm(m) for m in Sm]))
            loss_t = torch.mean(torch.stack([nuclear_norm(m) for m in Tm]))
            frob_s = torch.mean(torch.stack([frobenius_norm(fixed_s, m) for m in Sm]))
            frob_t = torch.mean(torch.stack([frobenius_norm(fixed_t, m) for m in Tm]))

        penalty = (loss_s + loss_t + F.relu(frob_s - args.delta) + F.relu(frob_t - args.delta)).item()

    # Evaluate accuracy on the batch
    tasks = []
    answers_true = []
    for rec in records:
        input_dict = {"task": rec["task"]}
        # mirror your original skip=True in dec stage
        tasks.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds, skip=is_dec)))
        answers_true.append(rec["answer"])

    raw = await asyncio.gather(*tasks)
    raw_answers, _log_probs = zip(*raw)

    correct = 0
    for ans, truth in zip(raw_answers, answers_true):
        pred = gsm_get_predict(ans[0])
        try:
            is_ok = float(pred) == float(truth)
        except Exception:
            is_ok = False
        correct += 1 if is_ok else 0

    accuracy = correct / max(1, len(records))
    fitness = accuracy - args.evo_reg * penalty
    return fitness, accuracy, penalty

async def _evo_optimize(
    graph: Graph,
    train_data: List[Dict[str, Any]],
    args,
    kwargs,
    agent_names: List[str],
    *,
    is_dec: bool,
    note: str
) -> Dict[str, TensorOrList]:
    """
    Generic EA loop. Returns best candidate params {'spatial': ..., 'temporal': ...}
    """
    pop_size = args.pop_size
    gens = args.evo_generations
    elite_k = max(1, int(pop_size * args.elite_fraction))
    eval_k = min(len(train_data), args.evo_eval_size)

    # Seed population from current graph params + noise
    def get_current_params():
        if is_dec:
            return {
                "spatial": _clone_param(graph.spatial_logits_1),
                "temporal": _clone_param(graph.temporal_logits_1),
            }
        else:
            return {
                "spatial": _clone_param(graph.spatial_logits),
                "temporal": _clone_param(graph.temporal_logits),
            }

    base = get_current_params()
    population = [base]
    for _ in range(pop_size - 1):
        cand = {
            "spatial": _mutate(_clone_param(base["spatial"]), args.mutation_rate, args.mutation_std),
            "temporal": _mutate(_clone_param(base["temporal"]), args.mutation_rate, args.mutation_std),
        }
        population.append(cand)

    best_overall = None
    best_fitness = -1e9
    history = []

    for g in range(gens):
        batch = random.sample(train_data, eval_k) if eval_k < len(train_data) else train_data
        scored = []
        # Evaluate each candidate
        for cand in population:
            fit, acc, pen = await _evaluate_candidate(graph, cand, batch, args, kwargs, agent_names, is_dec=is_dec)
            scored.append((fit, acc, pen, cand))
        scored.sort(key=lambda x: x[0], reverse=True)

        gen_best_fit, gen_best_acc, gen_best_pen, gen_best_cand = scored[0]
        history.append({
            "generation": g,
            "best_fitness": gen_best_fit,
            "best_accuracy": gen_best_acc,
            "penalty": gen_best_pen
        })

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_overall = {"spatial": _clone_param(gen_best_cand["spatial"]),
                            "temporal": _clone_param(gen_best_cand["temporal"])}

        # Elitism
        elites = [scored[i][3] for i in range(elite_k)]

        # Reproduce to refill population
        children = []
        while len(elites) + len(children) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = {
                "spatial": _uniform_crossover(p1["spatial"], p2["spatial"]),
                "temporal": _uniform_crossover(p1["temporal"], p2["temporal"])
            }
            child["spatial"] = _mutate(child["spatial"], args.mutation_rate, args.mutation_std)
            child["temporal"] = _mutate(child["temporal"], args.mutation_rate, args.mutation_std)
            children.append(child)

        population = elites + children

        # Progress log
        print(f"[EA:{note}] Gen {g+1}/{gens} | best_fitness={gen_best_fit:.4f} | acc={gen_best_acc:.3f} | penalty={gen_best_pen:.4f}")

    # Optional: write summary to result file
    try:
        result_dir = Path(f"{AgentPrune_ROOT}/result/MultiArith")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"EA_{note}_{Time.instance().value or 'time'}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"EA history write failed: {e}")

    return best_overall if best_overall is not None else base

# --------------------------
# Args
# --------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Experiments on gsm8k (EA-enabled)")
    parser.add_argument("--dataset_json", type=str, default="datasets/MultiArith/test.json")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="(unused when --evo) learning rate")
    parser.add_argument('--delta', type=float, default=0.1, help="noise level / structure margin")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5, help="(legacy) prune every few iterations")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning.")
    parser.add_argument('--num_iterations', type=int, default=10,help="(legacy) training iterations")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (dataset name)")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Agent names')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='Decision method of the agentprune')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--diff',action='store_true')
    parser.add_argument('--dec',action='store_true')
    parser.add_argument('--cot',action='store_true')

    # -------- EA switches --------
    parser.add_argument('--evo', action='store_true', help="Use evolutionary optimization instead of gradient-based", default=True)
    parser.add_argument('--pop_size', type=int, default=16)
    parser.add_argument('--evo_generations', type=int, default=8)
    parser.add_argument('--elite_fraction', type=float, default=0.25)
    parser.add_argument('--mutation_rate', type=float, default=0.10)
    parser.add_argument('--mutation_std', type=float, default=0.5)
    parser.add_argument('--evo_eval_size', type=int, default=10)
    parser.add_argument('--evo_reg', type=float, default=0.05, help="Weight for structure penalty in fitness")

    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args

# --------------------------
# Main
# --------------------------

async def main():
    args = parse_args()
    result_file = None
    dataset = JSONReader.parse_file(args.dataset_json)
    dataset = multiarith_data_process(dataset)
    train_dataset = JSONReader.parse_file('datasets/MultiArith/train.json')
    train_dataset = multiarith_data_process(train_dataset)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/MultiArith")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_llama3_{current_time}.json"
    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))

    with open(args.config_path, "r") as f:
        config = json.load(f)
        llm_list = list(config["model_list"].keys())

    graph = Graph(domain="gsm8k",
                    llm_list=llm_list,
                    agent_names=agent_names,
                    decision_method=decision_method,
                    optimized_spatial=args.optimized_spatial,
                    optimized_temporal=args.optimized_temporal,
                    rounds=args.num_rounds,
                    diff=args.diff,
                    dec=args.dec,
                    **kwargs)

    # --------------------------
    # EA Stage (DEC) — replaces Adam pretraining if requested
    # --------------------------
    # if args.dec and args.evo:
    #     # keep optimization flags off in this stage as in your original
    #     graph.optimized_spatial = False
    #     graph.optimized_temporal = False

    #     print("[EA] Starting DEC-stage evolutionary optimization...")
    #     best_dec = await _evo_optimize(
    #         graph, train_dataset, args, kwargs, agent_names, is_dec=True, note="DEC"
    #     )
    #     # set best params back to the main graph
    #     graph.spatial_logits_1 = best_dec["spatial"]
    #     graph.temporal_logits_1 = best_dec["temporal"]
    #     # Update masks as your original code does
    #     graph.update_masks_dec()
    #     print("[EA] DEC-stage finished and masks updated.")

    # --------------------------
    # EA Stage (Main) — replaces Adam + pruning loop if requested
    # --------------------------
    if args.evo and (args.optimized_spatial or args.optimized_temporal):
        graph.optimized_spatial = True
        graph.optimized_temporal = True

        print("[EA] Starting main evolutionary optimization...")
        best_main = await _evo_optimize(
            graph, train_dataset, args, kwargs, agent_names, is_dec=False, note="MAIN"
        )
        graph.spatial_logits = best_main["spatial"]
        graph.temporal_logits = best_main["temporal"]

        # After EA, you can apply a pruning step using your existing routines
        if args.pruning_rate > 0:
            if not graph.diff:
                spatial_masks, temporal_masks = graph.update_masks(args.pruning_rate)
            else:
                spatial_masks, temporal_masks = graph.update_masks_diff(args.pruning_rate)
            print("[EA] Applied pruning after EA.")
            if not graph.diff:
                print("spatial sparsity:", spatial_masks.sum()/spatial_masks.numel())
                print("temporal sparsity:", temporal_masks.sum()/temporal_masks.numel())
            else:
                print("spatial sparsity:", spatial_masks[0].sum()/spatial_masks[0].numel())
                print("temporal sparsity:", temporal_masks[0].sum()/temporal_masks[0].numel())

    # Clean up counters before evaluation
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    total_solved, total_executed = (0, 0)

    # --------------------------
    # Inference / evaluation loop (unchanged from your original bottom block)
    # --------------------------
    num_batches = int(len(dataset)/args.batch_size)
    for i_batch in range(num_batches):
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        add_losses = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        print(11111111)
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            
            if not graph.diff:
                spatial_matrix_train = realized_graph.spatial_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                temporal_matrix_train = realized_graph.temporal_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
            else:
                spatial_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.spatial_logits]
                temporal_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.temporal_logits]
            spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
            temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
            if not graph.diff:
                loss_s = nuclear_norm(spatial_matrix_train)
                loss_t = nuclear_norm(temporal_matrix_train)
                frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
            else:
                loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
            add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            
            task = record["task"]
            step = record["step"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
            add_losses.append(add_loss)
        
        print(22222222)
        raw_results = await asyncio.gather(*answer_log_probs)
        print(33333333)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        
        for task, answer, log_prob, add_loss, true_answer in zip(current_batch, raw_answers, log_probs, add_losses, answers):
            predict_answer = gsm_get_predict(answer[0])
            try:
                is_solved = float(predict_answer)==float(true_answer)
            except Exception:
                is_solved = False
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            utility = is_solved
            utilities.append(utility)
            # Note: we do not backprop here anymore; total_loss is informational
            single_loss = -log_prob * utility
            loss_list.append(single_loss+add_loss)
            updated_item = {
                "Question": task,
                "Answer": true_answer,
                "Step": step,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }
            data.append(updated_item)
            print(f"##########Final Log:{json.dumps(updated_item)}")
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

if __name__ == '__main__':
    asyncio.run(main())
