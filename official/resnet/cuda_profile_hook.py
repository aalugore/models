import tensorflow as tf
import ctypes

_cudart = ctypes.CDLL('libcudart.so')

class CudaProfileHook(tf.estimator.SessionRunHook):

  def begin(self):
    self._step_count = -1

  def before_run(self, run_context):
    self._step_count += 1
    if self._step_count == 60:
      print("BEGINNING PROFILING")
      _cudart.cudaProfilerStart()

  def after_run(self, run_context, run_values):
    if self._step_count == 70:
      print("ENDING PROFILING")
      _cudart.cudaProfilerStop()
