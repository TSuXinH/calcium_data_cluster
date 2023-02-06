from scipy.optimize import linear_sum_assignment
import numpy as np
import re
import matplotlib.pyplot as plt


def align_label(raw, next):
    length = np.max(raw).item() + 1
    G = np.zeros((length, length))
    for x in range(length):
        idx_raw = raw == x
        for y in range(length):
            idx_next = next == y
            G[x, y] = np.sum((idx_raw & idx_next).astype(np.int_))
    _, new_col = linear_sum_assignment(G)
    return new_col


def get_num(string):
    return [int(s) for s in re.findall(r'\d+', string)]

t1 = """361
360
358
357
357
356
352
351
348
341
329
310
257
179
"""
t2 = """
418
416
409
406
404
397
396
395
390
383
372
350
300
210
"""
t3 = """
473
470
468
467
466
463
463
460
459
454
445
427
355
271
"""
t4 = """
586
582
580
572
569
563
559
555
551
540
523
493
435
317
"""
t5 = """
591
589
584
576
569
556
553
550
542
524
513
485
408
301
"""
index = """
1
2 
5
10
20
50
70
100
200
500
700
1000
2000
5000
"""
t1i = get_num(t1)
t2i = get_num(t2)
t3i = get_num(t3)
t4i = get_num(t4)
t5i = get_num(t5)
ind = get_num(index)

plt.plot(ind, t1i, label='trial1')
plt.plot(ind, t2i, label='trial2')
plt.plot(ind, t3i, label='trial3')
plt.plot(ind, t4i, label='trial4')
plt.plot(ind, t5i, label='trial5')
plt.xticks(ind)
plt.title('Neurons picked corresponding to trials and thresholds')
plt.legend()
plt.show(block=True)

x = """198 199 211 224 261 296 301 364 365 372 374 383 422 465 520 523 536 543
 567 577 604 620 669 758 811"""
y = get_num(x)
print(len(y))