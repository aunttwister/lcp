"""LiveBench 2026-06-25 subtask leaderboard data (source: livebench.ai).

Shape: ``{benchmark_key: {category: {task_name: score_0_100}}}``. Sourced from
``table_2026_06_25.csv`` + ``categories_2026_06_25.json`` on livebench.ai.

Category names use LCP's canonical LiveBench category keys (matching
``seed_capabilities.LB_TO_LCP`` keys). Task names are lowercased and
underscored exactly as the ``all_tasks.csv`` columns produced by a LiveBench
run, so seeded subtasks and locally-run subtasks share the same shape.

NOTE: ``qwen3.6-27b-q4_k_m`` is a local GGUF quant; its scores are the
public ``qwen3.6-27b`` (fp16) leaderboard row as an approximation.
"""

LIVEBENCH_TASKS: dict[str, dict[str, dict[str, float]]] = {
    'deepseek-v4-pro': {
        "agentic_coding": {
            "javascript": 68.182,
            "python": 50.0,
            "typescript": 46.667
        },
        "coding": {
            "code_completion": 78.261,
            "code_generation": 76.056
        },
        "data_analysis": {
            "consecutive_events": 90.229,
            "tablejoin": 49.462,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 68.633,
            "simplify": 61.567,
            "story_generation": 74.933,
            "summarize": 65.667
        },
        "language": {
            "connections": 100.0,
            "plot_unscrambling": 64.222,
            "typos": 82.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 94.0,
            "math_comp": 97.059,
            "olympiad": 91.286
        },
        "reasoning": {
            "logic_with_navigation": 66.0,
            "spatial": 96.0,
            "theory_of_mind": 84.615,
            "zebra_puzzle": 96.75
        }
    },
    'deepseek-v4-flash': {
        "agentic_coding": {
            "javascript": 63.636,
            "python": 50.0,
            "typescript": 26.667
        },
        "coding": {
            "code_completion": 73.913,
            "code_generation": 76.056
        },
        "data_analysis": {
            "consecutive_events": 89.444,
            "tablejoin": 48.538,
            "tablereformat": 100.0
        },
        "instruction_following": {
            "paraphrase": 61.7,
            "simplify": 58.367,
            "story_generation": 70.5,
            "summarize": 71.5
        },
        "language": {
            "connections": 97.333,
            "plot_unscrambling": 58.192,
            "typos": 82.0
        },
        "math": {
            "amps_hard": 97.0,
            "integrals_with_game": 65.0,
            "math_comp": 96.078,
            "olympiad": 89.081
        },
        "reasoning": {
            "logic_with_navigation": 70.0,
            "spatial": 90.0,
            "theory_of_mind": 86.538,
            "zebra_puzzle": 100.0
        }
    },
    'claude-sonnet-5': {
        "agentic_coding": {
            "javascript": 68.182,
            "python": 60.0,
            "typescript": 50.0
        },
        "coding": {
            "code_completion": 78.261,
            "code_generation": 83.099
        },
        "data_analysis": {
            "consecutive_events": 74.741,
            "tablejoin": 42.442,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 61.25,
            "simplify": 60.65,
            "story_generation": 62.317,
            "summarize": 71.217
        },
        "language": {
            "connections": 95.5,
            "plot_unscrambling": 53.411,
            "typos": 76.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 88.0,
            "math_comp": 95.098,
            "olympiad": 90.671
        },
        "reasoning": {
            "logic_with_navigation": 86.0,
            "spatial": 100.0,
            "theory_of_mind": 80.769,
            "zebra_puzzle": 88.0
        }
    },
    'claude-fable-5': {
        "agentic_coding": {
            "javascript": 68.182,
            "python": 65.0,
            "typescript": 53.333
        },
        "coding": {
            "code_completion": 80.435,
            "code_generation": 91.549
        },
        "data_analysis": {
            "consecutive_events": 91.36,
            "tablejoin": 56.135,
            "tablereformat": 94.118
        },
        "instruction_following": {
            "paraphrase": 77.65,
            "simplify": 72.033,
            "story_generation": 71.7,
            "summarize": 81.7
        },
        "language": {
            "connections": 99.333,
            "plot_unscrambling": 78.719,
            "typos": 94.0
        },
        "math": {
            "amps_hard": 99.0,
            "integrals_with_game": 97.0,
            "math_comp": 95.098,
            "olympiad": 92.845
        },
        "reasoning": {
            "logic_with_navigation": 78.0,
            "spatial": 96.0,
            "theory_of_mind": 84.615,
            "zebra_puzzle": 100.0
        }
    },
    'claude-opus-5': {
        "agentic_coding": {
            "javascript": 77.273,
            "python": 75.0,
            "typescript": 43.333
        },
        "coding": {
            "code_completion": 82.609,
            "code_generation": 80.282
        },
        "data_analysis": {
            "consecutive_events": 77.571,
            "tablejoin": 51.962,
            "tablereformat": 94.118
        },
        "instruction_following": {
            "paraphrase": 65.733,
            "simplify": 61.583,
            "story_generation": 61.317,
            "summarize": 66.433
        },
        "language": {
            "connections": 99.333,
            "plot_unscrambling": 74.731,
            "typos": 92.0
        },
        "math": {
            "amps_hard": 99.01,
            "integrals_with_game": 97.0,
            "math_comp": 94.118,
            "olympiad": 92.796
        },
        "reasoning": {
            "logic_with_navigation": 86.0,
            "spatial": 100.0,
            "theory_of_mind": 78.846,
            "zebra_puzzle": 100.0
        }
    },
    'gpt-5.6-sol': {
        "agentic_coding": {
            "javascript": 63.636,
            "python": 55.0,
            "typescript": 50.0
        },
        "coding": {
            "code_completion": 84.783,
            "code_generation": 83.099
        },
        "data_analysis": {
            "consecutive_events": 90.214,
            "tablejoin": 49.308,
            "tablereformat": 100.0
        },
        "instruction_following": {
            "paraphrase": 71.517,
            "simplify": 70.517,
            "story_generation": 75.917,
            "summarize": 69.433
        },
        "language": {
            "connections": 100.0,
            "plot_unscrambling": 79.051,
            "typos": 84.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 100.0,
            "math_comp": 95.098,
            "olympiad": 91.693
        },
        "reasoning": {
            "logic_with_navigation": 82.0,
            "spatial": 100.0,
            "theory_of_mind": 84.615,
            "zebra_puzzle": 100.0
        }
    },
    'gpt-5.6-terra': {
        "agentic_coding": {
            "javascript": 68.182,
            "python": 50.0,
            "typescript": 46.667
        },
        "coding": {
            "code_completion": 80.435,
            "code_generation": 76.056
        },
        "data_analysis": {
            "consecutive_events": 87.67,
            "tablejoin": 50.25,
            "tablereformat": 100.0
        },
        "instruction_following": {
            "paraphrase": 57.033,
            "simplify": 61.433,
            "story_generation": 66.833,
            "summarize": 73.167
        },
        "language": {
            "connections": 93.0,
            "plot_unscrambling": 69.678,
            "typos": 86.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 95.0,
            "math_comp": 95.098,
            "olympiad": 91.535
        },
        "reasoning": {
            "logic_with_navigation": 76.0,
            "spatial": 100.0,
            "theory_of_mind": 86.538,
            "zebra_puzzle": 100.0
        }
    },
    'gpt-5.6-luna': {
        "agentic_coding": {
            "javascript": 63.636,
            "python": 45.0,
            "typescript": 36.667
        },
        "coding": {
            "code_completion": 86.957,
            "code_generation": 78.873
        },
        "data_analysis": {
            "consecutive_events": 86.504,
            "tablejoin": 47.596,
            "tablereformat": 100.0
        },
        "instruction_following": {
            "paraphrase": 47.817,
            "simplify": 58.667,
            "story_generation": 69.067,
            "summarize": 64.933
        },
        "language": {
            "connections": 96.5,
            "plot_unscrambling": 51.201,
            "typos": 70.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 70.0,
            "math_comp": 92.157,
            "olympiad": 88.646
        },
        "reasoning": {
            "logic_with_navigation": 80.0,
            "spatial": 96.0,
            "theory_of_mind": 73.077,
            "zebra_puzzle": 93.5
        }
    },
    'kimi-k3': {
        "agentic_coding": {
            "javascript": 68.182,
            "python": 65.0,
            "typescript": 53.333
        },
        "coding": {
            "code_completion": 82.609,
            "code_generation": 80.282
        },
        "data_analysis": {
            "consecutive_events": 89.758,
            "tablejoin": 48.404,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 74.367,
            "simplify": 66.367,
            "story_generation": 75.367,
            "summarize": 69.35
        },
        "language": {
            "connections": 100.0,
            "plot_unscrambling": 72.584,
            "typos": 84.0
        },
        "math": {
            "amps_hard": 97.0,
            "integrals_with_game": 54.0,
            "math_comp": 95.098,
            "olympiad": 91.649
        },
        "reasoning": {
            "logic_with_navigation": 80.0,
            "spatial": 100.0,
            "theory_of_mind": 82.692,
            "zebra_puzzle": 100.0
        }
    },
    'minimax-m3': {
        "agentic_coding": {
            "javascript": 63.636,
            "python": 35.0,
            "typescript": 23.333
        },
        "coding": {
            "code_completion": 67.391,
            "code_generation": 69.014
        },
        "data_analysis": {
            "consecutive_events": 84.614,
            "tablejoin": 45.846,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 55.933,
            "simplify": 56.617,
            "story_generation": 63.55,
            "summarize": 53.933
        },
        "language": {
            "connections": 96.333,
            "plot_unscrambling": 50.174,
            "typos": 84.0
        },
        "math": {
            "amps_hard": 75.0,
            "integrals_with_game": 56.0,
            "math_comp": 91.176,
            "olympiad": 85.614
        },
        "reasoning": {
            "logic_with_navigation": 66.0,
            "spatial": 94.0,
            "theory_of_mind": 76.923,
            "zebra_puzzle": 61.0
        }
    },
    'qwen3.8-max': {
        "agentic_coding": {
            "javascript": 77.273,
            "python": 60.0,
            "typescript": 56.667
        },
        "coding": {
            "code_completion": 73.913,
            "code_generation": 71.831
        },
        "data_analysis": {
            "consecutive_events": 87.142,
            "tablejoin": 50.058,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 73.317,
            "simplify": 67.35,
            "story_generation": 76.517,
            "summarize": 79.15
        },
        "language": {
            "connections": 94.5,
            "plot_unscrambling": 58.562,
            "typos": 86.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 81.0,
            "math_comp": 95.098,
            "olympiad": 91.151
        },
        "reasoning": {
            "logic_with_navigation": 74.0,
            "spatial": 100.0,
            "theory_of_mind": 78.846,
            "zebra_puzzle": 100.0
        }
    },
    'gemini-3.6-flash': {
        "agentic_coding": {
            "javascript": 63.636,
            "python": 40.0,
            "typescript": 26.667
        },
        "coding": {
            "code_completion": 78.261,
            "code_generation": 77.465
        },
        "data_analysis": {
            "consecutive_events": 43.824,
            "tablejoin": 47.135,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 74.317,
            "simplify": 75.7,
            "story_generation": 75.6,
            "summarize": 75.85
        },
        "language": {
            "connections": 100.0,
            "plot_unscrambling": 63.693,
            "typos": 88.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 63.0,
            "math_comp": 94.118,
            "olympiad": 90.492
        },
        "reasoning": {
            "logic_with_navigation": 76.0,
            "spatial": 100.0,
            "theory_of_mind": 78.846,
            "zebra_puzzle": 85.75
        }
    },
    'grok-4.5': {
        "agentic_coding": {
            "javascript": 72.727,
            "python": 60.0,
            "typescript": 36.667
        },
        "coding": {
            "code_completion": 69.565,
            "code_generation": 67.606
        },
        "data_analysis": {
            "consecutive_events": 75.536,
            "tablejoin": 43.577,
            "tablereformat": 100.0
        },
        "instruction_following": {
            "paraphrase": 69.217,
            "simplify": 70.167,
            "story_generation": 73.583,
            "summarize": 73.15
        },
        "language": {
            "connections": 98.0,
            "plot_unscrambling": 66.386,
            "typos": 84.0
        },
        "math": {
            "amps_hard": 99.0,
            "integrals_with_game": 78.0,
            "math_comp": 97.059,
            "olympiad": 89.239
        },
        "reasoning": {
            "logic_with_navigation": 72.0,
            "spatial": 100.0,
            "theory_of_mind": 82.692,
            "zebra_puzzle": 94.0
        }
    },
    'smaug-agentic': {
        "agentic_coding": {
            "javascript": 77.273,
            "python": 60.0,
            "typescript": 56.667
        },
        "coding": {
            "code_completion": 80.435,
            "code_generation": 84.507
        },
        "data_analysis": {
            "consecutive_events": 90.425,
            "tablejoin": 51.231,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 68.633,
            "simplify": 66.617,
            "story_generation": 76.667,
            "summarize": 72.05
        },
        "language": {
            "connections": 100.0,
            "plot_unscrambling": 71.07,
            "typos": 82.0
        },
        "math": {
            "amps_hard": 98.0,
            "integrals_with_game": 52.0,
            "math_comp": 94.118,
            "olympiad": 91.56
        },
        "reasoning": {
            "logic_with_navigation": 84.0,
            "spatial": 100.0,
            "theory_of_mind": 78.846,
            "zebra_puzzle": 98.25
        }
    },
    'qwen3.6-27b-q4_k_m': {
        "agentic_coding": {
            "javascript": 54.545,
            "python": 40.0,
            "typescript": 23.333
        },
        "coding": {
            "code_completion": 71.739,
            "code_generation": 71.831
        },
        "data_analysis": {
            "consecutive_events": 70.395,
            "tablejoin": 42.846,
            "tablereformat": 98.039
        },
        "instruction_following": {
            "paraphrase": 49.433,
            "simplify": 50.733,
            "story_generation": 60.133,
            "summarize": 52.617
        },
        "language": {
            "connections": 77.167,
            "plot_unscrambling": 42.746,
            "typos": 70.0
        },
        "math": {
            "amps_hard": 93.0,
            "integrals_with_game": 52.0,
            "math_comp": 92.157,
            "olympiad": 82.317
        },
        "reasoning": {
            "logic_with_navigation": 62.0,
            "spatial": 100.0,
            "theory_of_mind": 65.385,
            "zebra_puzzle": 53.75
        }
    },
}
