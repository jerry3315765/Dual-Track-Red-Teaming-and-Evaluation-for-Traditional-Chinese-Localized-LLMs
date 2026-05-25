"""
AutoDAN utilities for correlated-knowledge-jailbreak project.

This module contains the core AutoDAN implementation including:
- Genetic algorithm operations (crossover, mutation, selection)
- Hierarchical genetic algorithm (HGA) features
- Fitness evaluation based on model loss
- SuffixManager for conversation templates
- GPT mutation support
"""

from .opt_utils import (
    autodan_sample_control,
    autodan_sample_control_hga,
    get_score_autodan,
    get_score_autodan_low_memory,
    load_model_and_tokenizer,
    forward,
    roulette_wheel_selection,
    apply_crossover_and_mutation,
    crossover,
    gpt_mutate,
    apply_gpt_mutation,
    replace_with_synonyms,
    construct_momentum_word_dict,
    get_synonyms,
    word_roulette_wheel_selection,
    replace_with_best_synonym,
    apply_word_replacement,
)

from .string_utils import autodan_SuffixManager, load_conversation_template

__all__ = [
    "autodan_sample_control",
    "autodan_sample_control_hga",
    "get_score_autodan",
    "get_score_autodan_low_memory",
    "load_model_and_tokenizer",
    "forward",
    "roulette_wheel_selection",
    "apply_crossover_and_mutation",
    "crossover",
    "gpt_mutate",
    "apply_gpt_mutation",
    "replace_with_synonyms",
    "construct_momentum_word_dict",
    "get_synonyms",
    "word_roulette_wheel_selection",
    "replace_with_best_synonym",
    "apply_word_replacement",
    "autodan_SuffixManager",
    "load_conversation_template",
]
