import itertools

learning_rates = [0.0001, 0.0002, 0.0003, 0.0004]
amount_of_layers = [5, 8, 12]
latent_space = [200, 500, 1000]
loss_function = ['adam', 'sme']

if __name__ == "__main__":
    combinations: list = list(itertools.product(learning_rates, amount_of_layers, latent_space, loss_function))

    print(combinations)
