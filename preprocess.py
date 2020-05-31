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
