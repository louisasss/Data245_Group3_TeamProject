"""
Per-cluster identity data for the inference dashboard.
Z-scores are pulled from characterize_clusters.py output.
Example subreddits are hand-curated for recognizability.
"""

CLUSTER_PROFILES = {
    3: {
        0: {
            "label": "Mental Health & Trauma",
            "help": "The only strongly distinctive cluster. Subreddits here score above average on emotional entropy, multi-label count, and annotator disagreement — consistent with the layered, conflicting emotions typical of mental health and trauma communities. Examples: r/depression, r/SuicideWatch, r/BPD.",
            "z_scores": {
                "entropy": 0.26,
                "valence_mix": 0.09,
                "multi_label": 0.26,
                "disagreement": 0.22,
            },
            "examples": ["r/depression", "r/SuicideWatch", "r/BPD", "r/mentalhealth", "r/raisedbynarcissists"],
        },
        1: {
            "label": "Mixed (Group 1)",
            "help": "Slightly above-average on all four complexity features but no strong signature. Spans varied topics — Canadian regional discussion, sports (especially hockey), and dating advice — that share a similar emotional fingerprint despite being topically unrelated.",
            "z_scores": {
                "entropy": 0.05,
                "valence_mix": 0.05,
                "multi_label": 0.06,
                "disagreement": 0.05,
            },
            "examples": ["r/dating_advice", "r/vancouver", "r/politics", "r/canada", "r/EdmontonOilers"],
        },
        2: {
            "label": "Mixed (Group 2)",
            "help": "Slightly below-average on all four complexity features. A topically diverse cluster: men's-rights forums, reality TV gossip, fantasy football, ex-Mormon discussion. Different subjects, similar emotional shape.",
            "z_scores": {
                "entropy": -0.06,
                "valence_mix": -0.04,
                "multi_label": -0.07,
                "disagreement": -0.06,
            },
            "examples": ["r/MensRights", "r/90DayFiance", "r/confessions", "r/fantasyfootball", "r/exmormon", "r/worldpolitics"],
        },
    },
    4: {
        0: {
            "label": "Mixed (Group 1)",
            "help": "Weakly distinctive. K=4 splits one of the K=3 mainstream clusters into smaller groups — this one tends toward broad-interest discussion (politics, science, world news) alongside reality TV and confessional spaces.",
            "z_scores": {
                "entropy": 0.04,
                "valence_mix": 0.05,
                "multi_label": 0.03,
                "disagreement": 0.03,
            },
            "examples": ["r/MensRights", "r/90DayFiance", "r/confessions", "r/worldnews", "r/science"],
        },
        1: {
            "label": "Mixed (Group 2)",
            "help": "Weakly distinctive. Strongly Canadian-flavored: r/canada, r/CanadaPolitics, r/vancouver, r/EdmontonOilers, plus general advice and jobs subs. Topically coherent but emotionally similar to other clusters.",
            "z_scores": {
                "entropy": 0.04,
                "valence_mix": 0.03,
                "multi_label": 0.05,
                "disagreement": 0.05,
            },
            "examples": ["r/canada", "r/EdmontonOilers", "r/vancouver", "r/CanadaPolitics", "r/jobs"],
        },
        2: {
            "label": "Mixed (Group 3)",
            "help": "The most distinctive of the weak clusters: scores below average across all four complexity features. Spans fantasy football, ex-Mormon discussion, conservative politics, and New Zealand subreddits — different topics that share lower emotional complexity overall.",
            "z_scores": {
                "entropy": -0.11,
                "valence_mix": -0.09,
                "multi_label": -0.11,
                "disagreement": -0.10,
            },
            "examples": ["r/fantasyfootball", "r/exmormon", "r/worldpolitics", "r/Conservative", "r/newzealand"],
        },
        3: {
            "label": "Mental Health & Trauma",
            "help": "The only strongly distinctive cluster. Subreddits here score above average on emotional entropy, multi-label count, and annotator disagreement — consistent with the layered, conflicting emotions typical of mental health and trauma communities. Examples: r/depression, r/SuicideWatch, r/BPD.",
            "z_scores": {
                "entropy": 0.26,
                "valence_mix": 0.09,
                "multi_label": 0.26,
                "disagreement": 0.22,
            },
            "examples": ["r/depression", "r/SuicideWatch", "r/BPD", "r/mentalhealth", "r/raisedbynarcissists"],
        },
    },
}