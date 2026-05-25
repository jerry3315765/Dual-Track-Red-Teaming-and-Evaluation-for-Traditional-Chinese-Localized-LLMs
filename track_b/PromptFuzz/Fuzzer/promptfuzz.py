import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import json
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy, RoundRobinSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, NoMutatePolicy, MutateWeightedSamplingPolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.utils.predict import MatchPredictor, AccessGrantedPredictor
from gptfuzzer.llm import OpenAILLM, OpenAIEmbeddingLLM
from PromptFuzz.utils import constants

import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)


def run_fuzzer(args):
        
    from dotenv import load_dotenv
    track_b_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bundle_root = os.path.abspath(os.path.join(track_b_root, '..'))
    # Do not overwrite a runtime-injected key from shell/parent process.
    load_dotenv(os.path.join(bundle_root, '.env'), override=False)
    real_api_key = os.getenv('OPENAI_API_KEY') or getattr(args, 'openai_key', None)
    if not real_api_key or not real_api_key.startswith('sk-'):
        raise ValueError('OPENAI_API_KEY must be a real OpenAI API key for Track B; LM Studio or dummy values are not supported')
    results_root = getattr(args, 'results_root', os.path.join(bundle_root, 'results', 'track_b'))
    run_label = getattr(args, 'run_label', None) or getattr(args, 'selected_model_name', None) or getattr(args, 'baseline', None) or 'model'
    run_label = str(run_label).replace(os.sep, '_').replace('/', '_').replace(' ', '_')
    
    if args.mode == 'redteam':
        mutate_model = OpenAILLM('gpt-4o-mini', real_api_key)
    else:
        mutate_model = OpenAILLM(args.model_path, real_api_key)
        
    print(f"[INFO] Target Model: {args.model_path} @ {args.base_url}")
    target_model = OpenAILLM(args.model_path, real_api_key, base_url=args.base_url)

    if args.mode == 'hijacking':
        predictor = AccessGrantedPredictor()
    elif args.mode == 'extraction':
        predictor = MatchPredictor()
    elif args.mode == 'redteam':
        from gptfuzzer.utils.predict import LLMJudgePredictor
        judge_model_name = "gpt-4o-mini"
        print(f"[INFO] Using LLMJudgePredictor ({judge_model_name}) for classification.")
        predictor = LLMJudgePredictor(judge_model_name, real_api_key)
    else:
        raise ValueError("Invalid mode")
    
    if args.baseline != None:
        save_path = os.path.join(results_root, run_label, args.phase, args.mode, 'baseline', args.baseline, 'all_results.csv' if args.all_defenses else f'{args.index}.csv')
    else:
        save_path = os.path.join(results_root, run_label, args.phase, args.mode, 'all_results.csv' if args.all_defenses else f'{args.index}.csv')
    
    print("The save path is: ", save_path)

    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # load the defense prompt
    if args.phase == 'init':
        if args.mode == 'redteam':
            defense = os.path.join(track_b_root, 'Datasets', 'redteam_focus_defense.jsonl')
        elif args.baseline == 'human_expert' or args.baseline == 'gcg' or args.baseline == 'initial_seed':
            defense = os.path.join(track_b_root, 'Datasets', f'{args.mode}_focus_defense.jsonl')
        else:
            defense = os.path.join(track_b_root, 'Datasets', f'{args.mode}_robustness_dataset.jsonl')
    elif args.phase == 'preparation':
        defense = os.path.join(track_b_root, 'Datasets', f'{args.mode}_preparation_defense.jsonl')
    elif args.phase == 'focus':
        defense = os.path.join(track_b_root, 'Datasets', f'{args.mode}_focus_defense.jsonl')
    else:
        raise ValueError("Invalid phase")
    
    with open(defense, 'r', encoding='utf-8') as f:
        defenses = [json.loads(line) for line in f.readlines()]
    if args.all_defenses:
        args.defenses = defenses
    else:
        defenses = defenses[args.index]
        args.defenses = [defenses]
    print("The defense is: ", args.defenses)
    if args.no_mutate:
        assert args.phase == 'init'
    max_query_explicit = getattr(args, 'max_query_explicit', False)

    # load the initial seed
    if args.phase == 'init':
        if args.baseline == 'human_expert' or args.baseline == 'gcg':
            initial_seed_path = os.path.join(track_b_root, 'Datasets', f'{args.mode}_{args.baseline}_baseline.jsonl')
        elif args.baseline == 'initial_seed':
            initial_seed_path = os.path.join(track_b_root, 'Datasets', f'{args.mode}_preparation_seed.jsonl')
        else:
            initial_seed_path = os.path.join(track_b_root, 'Datasets', f'{args.mode}_robustness_dataset.jsonl')
    elif args.phase == 'focus':
        initial_seed_path = os.path.join(track_b_root, 'Datasets', f'{args.mode}_focus_seed.jsonl')
    elif args.phase == 'preparation':
        initial_seed_path = os.path.join(track_b_root, 'Datasets', f'{args.mode}_preparation_seed.jsonl')
        
    with open(initial_seed_path, 'r', encoding='utf-8') as f:
        initial_seed = [json.loads(line)['attack'] for line in f.readlines()]
    
    mutator_list = [
            OpenAIMutatorCrossOver(mutate_model, max_tokens=args.mutator_max_tokens),
            OpenAIMutatorExpand(mutate_model, max_tokens=args.mutator_max_tokens),
            OpenAIMutatorGenerateSimilar(mutate_model, max_tokens=args.mutator_max_tokens),
            OpenAIMutatorRephrase(mutate_model, max_tokens=args.mutator_max_tokens),
            OpenAIMutatorShorten(mutate_model, max_tokens=args.mutator_max_tokens)
            ]
    
    mutate_policy = MutateRandomSinglePolicy(
            mutator_list,
            concatentate=args.concatenate,
        )
    select_policy = MCTSExploreSelectPolicy()
    
    if args.no_mutate:
        mutate_policy = NoMutatePolicy()
        args.energy = 1
        # Cover all seeds across all defenses
        if not max_query_explicit:
            args.max_query = len(initial_seed) * len(args.defenses)
        select_policy = RoundRobinSelectPolicy()
        
    if args.phase == 'preparation':
        args.energy = 1
        if not max_query_explicit:
            args.max_query = len(initial_seed) * len(args.defenses) * 10
        select_policy = RoundRobinSelectPolicy()
        
    if args.phase == 'focus':
        args.energy = 5
        # args.max_query =  len(args.defenses) * 1000 # [MODIFIED] Use command line arg instead of hardcoded 14000
        select_policy = MCTSExploreSelectPolicy()
        
        few_shot_examples = pd.read_csv(os.path.join(track_b_root, 'Datasets', f'{args.mode}_few_shot_example.csv'))
        
        print(f"[INFO] Using embedding key (first 10 chars): {str(real_api_key)[:10]}...")
        embedding_model = OpenAIEmbeddingLLM("text-embedding-ada-002", real_api_key)
        
        mutate_policy = MutateWeightedSamplingPolicy(
            mutator_list,
            weights=args.mutator_weights,
            few_shot=args.few_shot,
            few_shot_num=args.few_shot_num,
            few_shot_file=few_shot_examples,
            concatentate=args.concatenate,
            retrieval_method=args.retrieval_method,
            cluster_num=args.cluster_num,
            embedding_model=embedding_model,
        )

    if args.max_query is None:
        args.max_query = 1000
        
    update_pool = True if args.phase == 'focus' else False
    
    fuzzer = GPTFuzzer(
        defenses=args.defenses,
        target=target_model,
        predictor=predictor,
        initial_seed=initial_seed,
        result_file=save_path,
        mutate_policy=mutate_policy,
        select_policy=select_policy,
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        target_max_tokens=args.target_max_tokens,
        update_pool=update_pool,
        dynamic_allocate=args.dynamic_allocate,
        threshold_coefficient=args.threshold_coefficient
    )

    fuzzer.run()
