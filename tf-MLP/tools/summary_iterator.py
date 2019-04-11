import tensorflow as tf

path_function = 'C:/Users/XOF/PycharmProjects/Training_MLP/tf-MLP/models/optimizer/quickdraw/gd/alfa0001'
path_events = 'events.out.tfevents.1554935222.DESKTOP-PMSD7B1'
with open('C:/Users/XOF/PycharmProjects/Training_MLP/loss/quickdraw_gd_0001.txt', 'a') as file:
    for e in tf.train.summary_iterator('{}/{}'.format(path_function, path_events)):
        for v in e.summary.value:
            if v.tag == 'loss':
                file.write('{}\n'.format(v.simple_value))
                # print(v.simple_value)
