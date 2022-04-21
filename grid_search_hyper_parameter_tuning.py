import itertools
from bayes_opt import BayesianOptimization

learning_rates = [0.0001, 0.0002, 0.0003, 0.0004]
amount_of_layers = [5, 8, 12]
latent_space = [200, 500, 1000]
loss_function = ['adam', 'sme']


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


if __name__ == "__main__":
    combinations: list = list(itertools.product(learning_rates, amount_of_layers, latent_space, loss_function))

    print(combinations)
    print(len(combinations))

    # Bounded region of parameter space
    pbounds = {'x': (2, 10), 'y': (-3, 13)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=3)

    print(optimizer.max)
