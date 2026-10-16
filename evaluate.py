"""
evaluate.py  |  Owner: SA  |  Interface: PROJECT_SCOPE.md §5
THE ONLY MODULE THAT SEES GROUND TRUTH (decision D4).

Scores a PosteriorSummary against the true labels and weights. Nothing
computed here may flow back into the sampler or into choosing which
state to report.
"""


def evaluate(summary, true_assignments, true_weights):
    """
    Args:
        summary:          PosteriorSummary
        true_assignments: (N,) true type of each trajectory
        true_weights:     list of K_true (F,) true weight vectors
    Returns:
        dict of metrics; keys are listed in the implementation's docstring
    """
    raise NotImplementedError
