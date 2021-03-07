
import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy')>=1):
			print("\nReached 99% accuracy so cancelling training!")
			self.model.stop_training = True
