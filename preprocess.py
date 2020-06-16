# -*- coding:utf-8 -*-


def text_process(text_contains_entities):
    text_contains_entities.encode('utf-8')
    raw_text = ''
    idx = 0
    true_idx = 0
    text_length = len(text_contains_entities)
    while idx < text_length:
        word = text_contains_entities[idx]
        temp = true_idx - 1
        digits = 1
        if temp == -1:
            digits = 1
        else:
            while temp // 10 != 0:
                digits += 1
                temp //= 10
        temp = true_idx
        digits_left = 1
        if temp == -1:
            digits_left = 1
        else:
            while temp // 10 != 0:
                digits_left += 1
                temp //= 10

        if word == '[' and idx < text_length - digits_left \
                and text_contains_entities[idx + 1:idx + 1 + digits_left] == str(true_idx):
            idx += 1
            idx += digits_left
            continue
        if word == '[':
            idx += 1
            continue
        if word == ' ' and idx + 1 < text_length and text_contains_entities[idx + 1] == '[':
            idx += 1
            continue
        if word == ' ':
            idx += 1
            continue
        if word == ']' and idx < text_length - 5 \
                and text_contains_entities[idx:idx + 5] in [']bod ', ']dis ', ']sym ', ']ite ']:
            idx += 5
            continue
        if word == ']' and idx < text_length - 4 \
                and text_contains_entities[idx:idx + 4] in [']bod', ']dis', ']sym', ']ite']:
            idx += 4
            continue
        if idx < text_length - digits and text_contains_entities[idx:idx + digits] == str(true_idx - 1) \
                and text_contains_entities[idx + digits:idx + digits + 5] in [']bod ', ']dis ', ']sym ', ']ite ']:
            idx += digits + 5
            continue
        if idx < text_length - digits and text_contains_entities[idx:idx + digits] == str(true_idx - 1) \
                and text_contains_entities[idx + digits:idx + digits + 4] in [']bod', ']dis', ']sym', ']ite']:
            idx += digits + 4
            continue
        raw_text += word
        true_idx += 1
        idx += 1
    return raw_text


def process_sym_attr(raw_text, attr):
    attr_dict = {}
    for k in attr:
        success = True
        if k == 'has_problem':
            continue
        k_list = [0] * len(raw_text)
        pos = attr[k]['pos']
        val = str(attr[k]['val']).split(' ')
        for idx, v in enumerate(val):
            start, end = None, None
            v_len = len(v)
            try:
                if v == '':
                    pass
                elif v == raw_text[pos[idx * 2]:pos[idx * 2 + 1] + 1]:
                    start, end = pos[idx * 2], pos[idx * 2 + 1] + 1
                elif v == raw_text[pos[idx * 2]:pos[idx * 2] + v_len]:
                    start, end = pos[idx * 2], pos[idx * 2] + v_len
                elif v == raw_text[pos[idx * 2 + 1] - v_len + 1:pos[idx * 2 + 1] + 1]:
                    start, end = pos[idx * 2 + 1] - v_len + 1, pos[idx * 2 + 1] + 1
                else:
                    start, end = pos[idx * 2], pos[idx * 2 + 1] + 1
                    success = False
            except:
                success = False

            if start is not None and end is not None:
                for index in range(start, end):
                    # if index == start:
                    #     k_list[index] = 'B'
                    # elif index == end - 1:
                    #     k_list[index] = 'E'
                    # else:
                    #     k_list[index] = 'I'
                    try:
                        k_list[index] = 1
                    except:
                        pass
                # if end - start <= 1:
                #     k_list[start] = 'S'
        attr_dict[k] = k_list
        # if not success:
        # print(text)
        # print('-' * 20)
        # print(filtered_text)
        # print('symtom:', symptom)
        # print('{}:{}'.format(k, success))
    return attr_dict


def is_number(s):
    try:
        float(s)
        return True

    except ValueError:
        pass
    return False


"""
return a dic ->(key,value)
key -> entity type ('dis','body','sym','ite')
value -> a list contains span pos([[0,2],[21,22],[34,37]...)
"""


def get_entity(text_contains_entities):
    text_contains_entities.encode('utf-8')
    stack = []
    entity_dic = {}
    idx = 0
    rel_pos = 0

    while idx < len(text_contains_entities):
        if text_contains_entities[idx] == '[':
            if stack == []:
                # 处理[]sym
                if text_contains_entities[idx + 1] == ']':
                    idx += 5
                    continue
                number_idx = idx + 1
                while is_number(text_contains_entities[number_idx]):
                    number_idx += 1
                true_idx = int(text_contains_entities[idx + 1:number_idx])
                stack.append(true_idx)
                rel_pos = true_idx
                idx += number_idx - idx
                continue
            else:
                stack.append(rel_pos)
                idx += 1
                continue

        elif text_contains_entities[idx] == ']':
            if len(stack) == 1:
                number_idx = idx - 1
                while is_number(text_contains_entities[number_idx]):
                    number_idx -= 1
                tail = int(text_contains_entities[number_idx + 1:idx])
                head = stack[-1]
                stack.pop()
                # if raw_text[idx+1:idx+4] in ['bod ', 'dis ', 'sym ', 'ite ']:
                entity_dic.setdefault(text_contains_entities[idx + 1:idx + 4], []).append([head, tail])
                idx += 4
                continue
            else:
                tail = rel_pos - 1
                head = stack[-1]
                stack.pop()
                entity_dic.setdefault(text_contains_entities[idx + 1:idx + 4], []).append([head, tail])
                idx += 4
                continue
        else:
            if text_contains_entities[idx] != ' ':
                rel_pos += 1
            idx += 1
            continue

    return entity_dic


"""
return a dic ->(key,value)
key -> entity type ('dis','body','sym','ite')
value -> a list(size=len(raw_text))  represent sequence labeling([0,1,1,1,0,0...])
"""


def get_entity_anno(text_contains_entities, raw_text_length):
    entity_annolist_dic = {}
    entity_dic = get_entity(text_contains_entities)

    for k in entity_dic.keys():
        array = [0] * raw_text_length
        for span in entity_dic[k]:
            array[span[0]:span[1] + 1] = [1] * (span[1] - span[0] + 1)
        entity_list_dic[k] = array

    return entity_annolist_dic


if __name__ == '__main__':
    text = '[0幽门肥厚性狭窄6]dis （ [8hypertrophicpyloricstenosis34]dis ， []dis ）是因 [42[幽门环状括约肌]bod 增厚50]sym [51[幽门管]bod 延长55]sym [56正常结构消失61]sym ，导致 [65[胃出口]bod 部梗阻70]sym [71[胃]bod 代偿性扩张 、 肥厚和蠕动加快84]sym [85[幽门平滑肌细胞]bod 肥大 ， 而非增生98]sym [99幽门肥厚性狭窄105]dis 的临床及病理表现，认为该病是一种先天性疾病。 胃空肠吻合术是当时经典的治疗方法，死亡率高达60%。 黏膜外幽门成形术是另一种术式，但因为缝合易撕裂水肿的 [180肌肉181]bod ，导致 [185大出血187]dis ，疗效也不理想。自从1911年Ramstedt首次放弃 缝合肌肉后， 幽门环肌切开术成为标准术式。'
    print(text)
    entity_dic = get_entity(text)
    print(entity_dic)
