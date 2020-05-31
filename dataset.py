import json

from torch.utils.data import Dataset


class MedicalExtractionDataset(Dataset):

    def __init__(self, data_path):
        super(MedicalExtractionDataset, self).__init__()
        self.data_path = data_path
        self.raw_data = None
        with open(file=data_path, mode='r', encoding='utf-8') as fr:
            self.raw_data = json.load(fr)

        self.data = None

    def __getitem__(self, item):

        pass

    def __len__(self):
        pass

    def pre_process(self, index):
        item = self.raw_data[index]
        text = item['text']
        symptoms = item['symptom']
        assert isinstance(symptoms, dict), 'dataset error: {}, index: {}'.format(self.data_path, index)

        for symtom_name, attribute_dict in symptoms:
            if attribute_dict['has_problem']:
                continue
            symtom_pos = None

            for attr_name, attr_property in attribute_dict:
                if attr_name == 'has_problem':
                    continue
                elif attr_name == 'self':
                    symtom_pos = attr_property['pos']
                else:
                    pass

    def text_process(self):
        pass
