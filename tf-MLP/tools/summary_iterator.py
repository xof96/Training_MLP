import tensorflow as tf

for e in tf.train.summary_iterator(
        'C:/Users/XOF/PycharmProjects/Training_MLP/tf-MLP/models/events.out.tfevents.1554760922.DESKTOP-PMSD7B1'):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)
