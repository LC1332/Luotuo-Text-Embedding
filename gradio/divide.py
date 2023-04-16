import torch


def divide_str(s, sep=['\n', '.', '。']):
    mid_len = len(s) // 2  # 中心点位置
    best_sep_pos = len(s) + 1  # 最接近中心点的分隔符位置
    best_sep = None  # 最接近中心点的分隔符
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  # 从中心点往左找分隔符
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos -
                                                        mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  # 没有找到分隔符
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]


def strong_divide(s):
    left, right = divide_str(s)

    if right != '':
        return left, right

    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',\
                 '：', '！', '？', '(', ')', '”', '“', \
                 '’', '‘', '[', ']', '{', '}', '<', '>', \
                 '/', '''\''', '|', '-', '=', '+', '*', '%', \
               '$', '''#''', '@', '&', '^', '_', '`', '~',\
                 '·', '…']
    left, right = divide_str(s, sep=whole_sep)

    if right != '':
        return left, right

    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]


def divide_inputs(input_text):
    lefts = []
    rights = []
    for text in input_text:
        left, right = strong_divide(text)
        lefts.append(left)
        rights.append(right)

    return lefts, rights


def get_embedding(model, tokenizer, inputs):
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs,
                           output_hidden_states=True,
                           return_dict=True,
                           sent_emb=True).pooler_output

    return embeddings


def get_all_embedding(model, tokenizer, text_left,text_right):
    embeddings_left = get_embedding(model, tokenizer, text_left)
    embeddings_right = get_embedding(model, tokenizer, text_right)
    return embeddings_left, embeddings_right