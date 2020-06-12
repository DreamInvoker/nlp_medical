# -*- coding:utf-8 -*-
import re
ent_map={
    "bod":1,
    "dis":2,
    "sym":3,
    "ite":4
}


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
        k_list = []
        pos = attr[k]['pos']
        val = str(attr[k]['val']).split(' ')
        for idx, v in enumerate(val):
            start, end = None, None
            v_len = len(v)
            try:
                if v == '':
                    pass
                else:
                    start, end= pos[idx*2], pos[2*idx+1]
                    success = False
                '''
                elif v == raw_text[pos[idx * 2]:pos[idx * 2 + 1] + 1]:
                    start, end = pos[idx * 2], pos[idx * 2 + 1] + 1
                elif v == raw_text[pos[idx * 2]:pos[idx * 2] + v_len]:
                    start, end = pos[idx * 2], pos[idx * 2] + v_len
                elif v == raw_text[pos[idx * 2 + 1] - v_len + 1:pos[idx * 2 + 1] + 1]:
                    start, end = pos[idx * 2 + 1] - v_len + 1, pos[idx * 2 + 1] + 1
                '''
           
            except:
                success = False

            if start is not None and end is not None:
                k_list.append(start)
                k_list.append(end)
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
    stack =[]
    entity_dic ={}
    idx = 0
    rel_pos = 0

    while idx < len(text_contains_entities):
        if text_contains_entities[idx]=='[':
            if stack==[]:
                number_idx = idx+1
                while is_number(text_contains_entities[number_idx]):
                    number_idx+=1
                true_idx =int(text_contains_entities[idx+1:number_idx])
                stack.append(true_idx)
                rel_pos=true_idx
                idx+=number_idx-idx
                continue
            else:
                stack.append(rel_pos)
                idx+=1
                continue

        elif text_contains_entities[idx]==']':
            if len(stack) == 1:
                number_idx = idx-1
                while is_number(text_contains_entities[number_idx]):
                    number_idx-=1
                tail=int(text_contains_entities[number_idx+1:idx])
                head=stack[-1]
                stack.pop()
                #if raw_text[idx+1:idx+4] in ['bod ', 'dis ', 'sym ', 'ite ']:
                entity_dic.setdefault(text_contains_entities[idx+1:idx+4], []).append([head,tail])
                idx+=4
                continue
            else:
                tail = rel_pos-1
                head = stack[-1]
                stack.pop()
                entity_dic.setdefault(text_contains_entities[idx + 1:idx + 4], []).append([head, tail])
                idx += 4
                continue
        else:
            if text_contains_entities[idx] !=' ':
                rel_pos+=1
            idx += 1
            continue

    return entity_dic



def getnooverlap_entity(text):
    i=0
    entity_list=[]
    success=True
    while i<len(text):
        if text[i]=='[' :
            stack=[]
            j=i+1
            while j<len(text):
                if text[j]==']' and len(stack)==0:
                    break;
                elif text[j]==']' and len(stack)>0:
                    stack.pop()
                    j+=1
                elif text[j]=='[':
                    stack.append(text[j])
                    j+=1

                else:
                    j+=1
            if j==i+1:
                i=j+1
                continue

            entspan=text[i:j+1]
            str1=re.findall("\d+",entspan)[0]
            str2=re.findall("\d+",entspan)[-1]
            x0,y0=int(str1),int(str2)
            if abs(abs(j-i)-abs(x0-y0))<40:
                start,end=x0,y0
            elif  len(str1)<len(str2):
                str20=str2[-1*len(str1):]
                start,end=x0,int(str20)
                if start>end:
                    end=int(str2[-1*(len(str1)+1):])
            else:
                str10=str1[:len(str2)]
                start,end=int(str10),y0
                if start>end:
                    start=int(str1[:len(str2)-1])
            
            if start>1000 or end>1000:
                success=False
            
            enttype=ent_map[text[j+1:j+4]]
            entity_list.append([start,end,enttype])
            i=j+1
        else:
            i+=1

    return entity_list,success





"""
return a dic ->(key,value)
key -> entity type ('dis','body','sym','ite')
value -> a list(size=len(raw_text))  represent sequence labeling([0,1,1,1,0,0...])
"""
def get_entity_anno(text_contains_entities,raw_text_length):
    entity_annolist_dic ={}
    entity_dic = get_entity(text_contains_entities)

    for k in entity_dic.keys():
        array =[0]*raw_text_length
        for span in entity_dic[k]:
            array[span[0]:span[1]+1]=[1]*(span[1]-span[0]+1)
        entity_list_dic[k]=array

    return entity_annolist_dic
