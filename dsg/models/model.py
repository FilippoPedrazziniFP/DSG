class Model(object):
    def __init__(self, epochs=300, steps_per_epoch=1, hyperp=None):
        super(BiLSTMCNN, self).__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.load_embeddings_and_parameters()
        self.hyper_params = self.create_hp_dictionary(hyperp)
        self.model = None
        self.label_dim = None