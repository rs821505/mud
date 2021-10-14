from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from mud.base import IterativeLinearProblem
from mud.util import std_from_equipment


def rotation_map(qnum=10, tol=0.1, b=None, ref_param=None, seed=None):
    """
    Generate test data linear rotation map

    """
    if seed is not None:
        np.random.seed(24)

    vec = np.linspace(0, np.pi, qnum)
    A = np.array([[np.sin(theta), np.cos(theta)] for theta in vec])
    A = A.reshape(qnum, 2)
    b = np.zeros((qnum,1)) if b is None else b
    ref_param = np.array([[0.5, 0.5]]).reshape(-1,1) if ref_param \
            is None else ref_param

    # Compute observed value
    y = A@ref_param + b
    initial_mean = np.random.randn(2).reshape(-1,1)
    initial_cov = np.eye(2)*std_from_equipment(tol)

    return (A, b, y, initial_mean, initial_cov, ref_param)


def rotation_map_trials(numQoI=10, method='ordered', num_trials=100,
                        model_eval_budget=100, ax=None, color='r',
                        label='Ordered QoI $(10\\times 10D)$', seed=None):
    """
    Run a set of trials for linear rotation map problems

    """

    # Initialize plot if axis object is not passed in
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=(20,10))

    # Build Rotation Map. This will initialize seed of trial if specified
    A, b, y, initial_mean, initial_cov, ref_param = rotation_map(qnum=numQoI, seed=seed)

    # Calcluate number of epochs to use per trial based off of model budget and number of QoI
    num_epochs = model_eval_budget//numQoI

    errors = []
    for trial in range(num_trials):
        # Get a new random initial mean to start from per trial on same problem
        initial_mean = np.random.rand(2,1)

        # Initialize number of epochs and idx choices to use on this trial
        epochs = num_epochs
        choice = np.arange(numQoI)

        # Modify epochs/choices based off of method
        if method=='ordered':
            # Ordered - Go through each row of A in order once per epoch, same per trial
            epochs = num_epochs
        elif method=='shuffle':
            # Shuffled - Shuffle rows of A on each trial, go in order once per epoch.
            np.random.shuffle(choice)
            epochs = num_epochs
        elif method=='batch':
            # Batch - Perform only one epoch, but iterate in random order num_epochs times over each row of A
            choice = list(np.arange(numQoI))*num_epochs
            np.random.shuffle(choice)
            epochs = 1
        elif method=='random':
            # Randoms - Perform only one epoch, but do num_epochs*rows random choices of rows of A, with replacement.
            choice = np.random.choice(np.arange(numQoI), size=num_epochs*numQoI)
            epochs = 1

        # Initialize Iterative Linear Problem and solve using number of epochs
        prob = IterativeLinearProblem(A, b=b, y=y, initial_mean=initial_mean, cov=initial_cov, idx_order=choice)
        _ = prob.solve(num_epochs=epochs)

        # Plot errors with respect to reference parameter over each iteration
        prob.plot_chain_error(ref_param, alpha=0.1, ax=ax, color=color, fontsize=36)

        # Append to erros matrix to calculate mean error accross trials
        errors.append(prob.get_errors(ref_param))

    # Compute mean errors at each iteration across all trials
    avg_errs = np.mean(np.array(errors),axis=0)

    # Plot mean errors
    ax.plot(avg_errs, color, lw=5, label=label)

