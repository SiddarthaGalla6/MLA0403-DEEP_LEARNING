# CODE:
import tensorflow as tf
import matplotlib.pyplot as plt
(x,_),_=tf.keras.datasets.mnist.load_data(); x=x[:6000].astype("float32")/255.; x=x.reshape(-1,784)
latent=16
enc_in=tf.keras.Input((784,)); h=tf.keras.layers.Dense(128,activation="relu")(enc_in)
mu=tf.keras.layers.Dense(latent)(h); logvar=tf.keras.layers.Dense(latent)(h)
def sampling(args):
 mu,lv=args; eps=tf.random.normal(tf.shape(mu)); return mu+tf.exp(.5*lv)*eps
z=tf.keras.layers.Lambda(sampling)([mu,logvar]); enc=tf.keras.Model(enc_in,[mu,logvar,z])
dec_in=tf.keras.Input((latent,)); d=tf.keras.layers.Dense(128,activation="relu")(dec_in); d=tf.keras.layers.Dense(784,activation="sigmoid")(d); dec=tf.keras.Model(dec_in,d)
class VAE(tf.keras.Model):
 def train_step(self,data):
  if isinstance(data,tuple): data=data[0]
  with tf.GradientTape() as tape:
   m,l,z=enc(data); rec=dec(z); bce=tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data,rec),axis=-1))
   kl=-.5*tf.reduce_mean(tf.reduce_sum(1+l-tf.square(m)-tf.exp(l),axis=-1)); loss=bce+kl
  vars=enc.trainable_variables+dec.trainable_variables; grads=tape.gradient(loss,vars); self.optimizer.apply_gradients(zip(grads,vars)); return {"loss":loss}
vae=VAE(); vae.compile(optimizer="adam"); vae.fit(x,epochs=2,batch_size=128,verbose=0)
samples=dec(tf.random.normal((10,latent))).numpy().reshape(-1,28,28)
fig,ax=plt.subplots(1,10,figsize=(12,2))
for i in range(10): ax[i].imshow(samples[i],cmap="gray"); ax[i].axis("off")
plt.suptitle("VAE-generated samples"); plt.savefig("29_vae_gan_sample_generation.png"); plt.show()
print("Generated 10 samples (VAE demonstration; GAN is a separate model family).")

# INPUT:
# Input: MNIST training images; random latent vectors.
#

# OUTPUT:
# Output: grid of generated VAE samples. This script demonstrates VAE sampling, not a trained GAN.
