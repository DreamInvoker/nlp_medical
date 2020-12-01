import os
import json

with open('body_output.json', 'r') as fr:
    body_output = json.load(fr)

with open('output.json', 'r') as fr:
    output = json.load(fr)

fw = open('cases.txt', 'w')

attr_list = ['subject', 'body', 'freq', 'decorate']

for body_dict, dic in zip(body_output, output):
    dic.update(body_dict)
    raw_text = ''.join(dic['text'])
    symptom_name = ''.join(dic['symptom_name'])
    fw.write('text:{}\n'.format(raw_text))
    fw.write('symptom:{}\n'.format(symptom_name))

    for attr in attr_list:
        predict = dic['predict_'+attr]
        gold = dic['gold_'+attr]
        fw.write('{} pred:{}, gold:{}\n'.format(attr, ' '.join(predict), ' '.join(gold)))
    fw.write('*'*30)
    fw.write('\n')


fw.close()