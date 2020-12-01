# statistic

| data      | # passage    | # no problem symptoms | total feature
| ---------- | :-----------:  | :-----------: | :-----------: |
| train     | 4227 (78.7%)     |  17268 (79.7%)   | -     |
| dev     | 318 (5.9%)    |      1167 (5.4%)    |-     |
| test     | 824  (15.3%)   |    3218 (14.9%)    |-     |

# manually process attributes with wrong label

| data      | number    | pos number | percentage     | is solved |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: |
| train     | 472     | 148|  -  | no |
| dev     | 27     |  0|  -  | yes
| test     | 186     |  85 | -    | no |


# format
```
[
    {
        "text": "xxx",  // 原始段落（未分词）
        "symptom": { // 该段落中所有的症状
          "No.index: 症状实体名称": {
            "has_problem": false, // 该是否有问题
            "self": { // 症状实体本身
              "val": "",
              "pos": [start_index, end_index],                
            },
            "subject": {
              "val": "",
              "pos": [start_index, end_index],
            },
            {其他属性}
          },
          {其他症状实体}
        }
    },
    {其他段落}
]
```
