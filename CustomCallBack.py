
import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy')>0.90):
			print("\nReached 90% accuracy so cancelling training!")
			self.model.stop_training = True
