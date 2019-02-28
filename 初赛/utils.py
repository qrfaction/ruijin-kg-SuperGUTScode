from contextlib import contextmanager


@contextmanager
def timer(name):
    import time
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} s'.format(name, int(elapsedTime)))


def F1_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    result_pred = set()
    result_true = set()
    for i, (a, b) in enumerate(zip(y_true, y_pred)):
        idx = a > -1
        x = a[idx]
        y = b[idx]
        s = 0
        e = 1

        while e <= len(x):
            if x[s] == 0:
                s += 1
                e += 1
            elif e < len(x) and x[e] != 0:
                e += 1
            else:
                result_true.add((i, s, e, tuple(x[s:e])))
                s = e
                e += 1

        s = 0
        e = 1
        while e <= len(y):
            if y[s] == 0:
                s += 1
                e += 1
            elif e < len(y) and y[e] != 0:
                e += 1
            else:
                result_pred.add((i, s, e, tuple(y[s:e])))
                s = e
                e += 1

    tp = len(result_pred & result_true)
    f1 = 2 * tp / (len(result_pred) + len(result_true))
    return f1


def F1_score_v2(y_true, y_pred, index):
    assert y_true.shape == y_pred.shape
    result_pred = set()
    result_true = set()
    for i, (a, b, (idx1, idx2)) in enumerate(zip(y_true, y_pred, index)):
        idx = a > -1
        x = a[idx][idx1:idx2]
        y = b[idx][idx1:idx2]
        s = 0
        e = 1

        while e <= len(x):
            if x[s] == 0:
                s += 1
                e += 1
            elif e < len(x) and x[e] != 0:
                e += 1
            else:
                result_true.add((i, s, e, tuple(x[s:e])))
                s = e
                e += 1

        s = 0
        e = 1
        while e <= len(y):
            if y[s] == 0:
                s += 1
                e += 1
            elif e < len(y) and y[e] != 0:
                e += 1
            else:
                result_pred.add((i, s, e, tuple(y[s:e])))
                s = e
                e += 1

    tp = len(result_pred & result_true)
    f1 = 2 * tp / (len(result_pred) + len(result_true))
    return f1

def F1_score_v3(y_true, y_pred, index):
    assert y_true.shape == y_pred.shape
    result_pred = set()
    result_true = set()
    for i, (a, b, (idx1, idx2)) in enumerate(zip(y_true, y_pred, index)):
        idx = a > -1
        x = a[idx][idx1:idx2]
        y = b[idx][idx1:idx2]
        s = 0
        e = 1

        while e <= len(x):
            if x[s] == 0:
                s += 1
                e += 1
            elif e < len(x) and x[e] != 0:
                e += 1
            else:
                result_true.add((i, s, e, x[s]))
                s = e
                e += 1

        s = 0
        e = 1
        while e <= len(y):
            if y[s] == 0:
                s += 1
                e += 1
            elif e < len(y) and y[e] != 0:
                e += 1
            else:
                result_pred.add((i, s, e, y[s]))
                s = e
                e += 1

    tp = 0
    for x in result_true:
        for y in result_pred:
            if x[0]!=y[0] or x[3]!=y[3]:
                continue
            if max(x[1],y[1]) <= min(x[2],y[2]):
                tp += 1
    f1 = 2 * tp / (len(result_pred) + len(result_true))
    return f1


def split_text(text):
    split_chars = set('。，,;')

    def is_chinese(ch):
        if '\u4e00' <= ch <= '\u9fff':
            return True
        return False

    import re
    split_idx = []

    # r = [(m.group(), m.span())for ]
    for m in re.finditer('。|，|,|;|\.|』', text):

        idx = m.span()[0]
        if text[idx - 1] == '\n':
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isdigit():
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isspace() and text[idx + 2].isdigit():
            continue
        if text[idx - 1].islower() and text[idx + 1].islower():
            continue
        if text[idx + 1] in split_chars:
            continue
        if text[idx] == '.':
            if re.search('\d\s+\.\s+\d', text[idx - 2:idx + 2]):
                continue
            if text[idx + 1:idx + 4] == 'com':
                continue

        split_idx.append(idx + 1)



    pattern = '\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'
    pattern += '注:|附录 |表 \d|Tab \d+|\[摘要\]|\[关键词\]|\[提要\]|表\d[^。，,;]+?\n|图 \d |Fig \d |'
    pattern += '\[Abstract\]|\[Summary\]|前    言|【摘要】|【关键词】|结    果|讨    论|'
    pattern += '【Sunnnary】|and |with |or |by |because of |as well as '
    for m in re.finditer(pattern, text):
        idx = m.span()[0]
        if text[idx].isupper() and idx > 0:
            i = idx - 1
            while i > 0 and text[i] == ' ':
                i -= 1
            if text[i] not in ['\n'] and is_chinese(text[i]) == False:
                continue
            idx = i + 1
        if (text[idx:idx + 2] in ['or', 'by'] or text[idx:idx + 3] == 'and' or text[idx:idx + 4] == 'with') \
                and (text[idx - 1].islower() or text[idx - 1].isupper()):
            continue

        split_idx.append(idx)

    for m in re.finditer('\n\(\d\)', text):
        idx = m.span()[0]+1
        split_idx.append(idx)

    split_idx = list(sorted(set([0, len(text)] + split_idx)))
    other_idx = []
    for i in range(len(split_idx) - 1):
        b = split_idx[i]
        e = split_idx[i + 1]
        if text[b] in '一二三四五六七八九零十':
            for j in range(b, e):
                if text[j] == '\n':
                    other_idx.append(j + 1)
                    break
    split_idx += other_idx
    split_idx = list(sorted(set([0, len(text)] + split_idx)))


    other_idx = []
    for i in range(len(split_idx) - 1):
        b = split_idx[i]
        e = split_idx[i + 1]

        other_idx.append(b)
        if e - b > 160:
            for j in range(b, e):
                if (j+1-other_idx[-1])>15:

                    if text[j] == '\n':
                        other_idx.append(j + 1)
                    if text[j] == ' ' and text[j-1].isnumeric() and text[j+1].isnumeric():
                        other_idx.append(j + 1)
    split_idx += other_idx
    split_idx = list(sorted(set([0, len(text)] + split_idx)))


    for i in range(1,len(split_idx)):
        idx = split_idx[i]
        while idx > split_idx[i-1]-1 and text[idx-1].isspace():
            idx -= 1
        split_idx[i] = idx

    split_idx = list(sorted(set([0, len(text)] + split_idx)))

    temp_idx = []
    i = 0
    while i < len(split_idx) - 1:
        b = split_idx[i]
        e = split_idx[i + 1]

        num_ch = 0
        num_en = 0
        if e - b < 15:
            for w in text[b:e]:
                if is_chinese(w):
                    num_ch += 1
                elif w.islower() or w.isupper():
                    num_en += 1
                if num_ch + 0.5 * num_en > 5:
                    temp_idx.append(b)
                    break
            if num_ch + 0.5 * num_en <= 5:
                if i==0:
                    assert b==0
                    temp_idx.append(b)
                    i += 2
                elif i

        else:
            temp_idx.append(b)
            i += 1
    split_idx = list(sorted(set([0, len(text)] + temp_idx)))

    results = []
    for i in range(len(split_idx) - 1):
        b = split_idx[i]
        e = split_idx[i + 1]
        results.append(text[b:e])
    s = ''
    for r in results:
        s += r
    assert len(s) == len(text)

    return results


if __name__ == '__main__':

    import os
    num_s = 0
    samples = []
    ids = set([x.split('.')[0] for x in os.listdir('../data/train/')])
    for idx in sorted(ids):
        with open(f'../data/train/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_text(texts)  # 自定义分割方法
        num_s += len(texts)
        # for s in texts:
        #     print(s.replace('\n','_LB_'),'\n')
        # print(idx)
        samples += texts


    print('*'*50)
    for s in samples:
        if len(s)>160:
            print(s.replace('\n','_LB_'),'\n')

    print(num_s)
    # ids = set([x.split('.')[0] for x in os.listdir('../data/test/')])
    # for idx in sorted(ids):
    #     with open(f'../data/test/{idx}.txt', 'r', encoding='utf-8') as f:
    #         texts = f.read()
    #         texts = split_text(texts)  # 自定义分割方法
        # for s in texts:
        #     print(s.replace('\n','_LB_'),'\n')
        # print(idx)
        # break


















