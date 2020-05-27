import json

from torch.utils.data import Dataset



class MedicalExtractionDataset(Dataset):

    def __init__(self, data_path):
        super(MedicalExtractionDataset, self).__init__()
        self.data = None
        with open(file=data_path, mode='r', encoding='utf-8') as fr:
            self.data = json.load(fr)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def pre_process(self):
        pass
