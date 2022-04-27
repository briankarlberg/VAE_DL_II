import itertools
from bayes_opt import BayesianOptimization
from library.three_encoder_vae.three_encoder_architecture import ThreeEncoderArchitecture
from functools import partial

learning_rates = [0.0001, 0.0002, 0.0003, 0.0004]
amount_of_layers = [5, 8, 12]
latent_space = [200, 500, 1000]
loss_function = ['adam', 'sme']


if __name__ == "__main__":


    # define constants during run time
    fit_with_partial = partial(ThreeEncoderArchitecture.build_three_variational_auto_encoder, x_train, y_train, x_test,
                               y_test)

    # Bounded region of parameter space
    boundaries = {'x': (5.0, 10), 'y': (-3, 13)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=boundaries,
        random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=3)

    print(optimizer.max)
