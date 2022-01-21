'''
Classe abstrata DataLoader. Modelo para criação de clases para gerenciar datasets.
'''
class DataLoader(object):
    def __init__(self):
        pass

    def download_data(self):
        raise NotImplementedError

    