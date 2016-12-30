import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '/home/zhipeng/Work/DL/caffemodels/fcn8s-heavy-pascal.caffemodel'

# init
caffe.set_device(2)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/zhipeng/Work/DL/caffe/fcn/lidc-fcn8s/solver_test.prototxt')
solver.net.copy_from(weights)

# solver.net.save('/home/zhipeng/Work/DL/caffemodels/lidc-fcn8s-tmp.caffemodel')
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

val = np.loadtxt('/home/zhipeng/Work/Data/LIDC-annotation-val/lidc_test.txt', dtype=str)
# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)

for _ in range(25):
   solver.step(40)
   score.seg_tests(solver, False, val, layer='score')

solver.test_nets[0].share_with(solver.net)
# print solver.test_nets[0]
# print solver.test_nets[0].blobs['score'].channels
# print solver.test_nets[0].blobs['label'].data[0,0].flatten()
# print solver.test_nets[0].blobs['score'].data[0].argmax(0)
solver.net.save('/home/zhipeng/Work/DL/caffemodels/lidc-fcn8s-test.caffemodel')
