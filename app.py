import rembg
from rembg import remove

import streamlit as st

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import PIL
from PIL import Image
import random

def go_crazy(input_image = "./donald_15K.jpg",octave_scale=1.4):

  # Normalize an image
  def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

#   # Display an image
#   def show(img):
#     display.display(PIL.Image.fromarray(np.array(img)))

  # Downsizing the image makes it easier to work with.
  base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

  # Maximize the activations of these layers
  names = ['mixed3', 'mixed5']
  layers = [base_model.get_layer(name).output for name in names]

  # Create the feature extraction model
  dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

  def calc_loss(img, model_a):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model_a(img_batch)
    if len(layer_activations) == 1:
      layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
      loss = tf.math.reduce_mean(act)
      losses.append(loss)

    return  tf.reduce_sum(losses)

  class DeepDream(tf.Module):
    def __init__(self, model_b):
      self.modelb = model_b

    @tf.function(
        input_signature=(
          tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
          tf.TensorSpec(shape=[], dtype=tf.int32),
          tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
          with tf.GradientTape() as tape:
            # This needs gradients relative to `img`
            # `GradientTape` only watches `tf.Variable`s by default
            tape.watch(img)
            loss = calc_loss(img, self.modelb)

          # Calculate the gradient of the loss with respect to the pixels of the input image.
          gradients = tape.gradient(loss, img)

          # Normalize the gradients.
          gradients /= tf.math.reduce_std(gradients) + 1e-8

          # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
          # You can update the image by directly adding the gradients (because they're the same shape!)
          img = img + gradients*step_size
          img = tf.clip_by_value(img, -1, 1)

        return loss, img

  deepdream = DeepDream(dream_model)

  def run_deep_dream_simple(img, steps=1, step_size=0.04):
    # Convert from uint8 to the range expected by the model.

    #img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.image.convert_image_dtype(img, tf.float32) #added
    img = (img - 127.5) / 127.5
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
      if steps_remaining>100:
        run_steps = tf.constant(100)
      else:
        run_steps = tf.constant(steps_remaining)
      steps_remaining -= run_steps
      step += run_steps

      loss, img = deepdream(img, run_steps, tf.constant(step_size))

      #display.clear_output(wait=True)
      #show(deprocess(img))
      #print ("Step {}, loss {}".format(step, loss))


    result = deprocess(img)
    #display.clear_output(wait=True)
    #show(result)

    return result

  #if input image is a string, then we assume it is a path
  if type(input_image) == str:
    input = Image.open(input_image).convert("RGBA")
    input = remove(input)
  else: #if it's not a string then we assume it is already a picture and take it as such
    pil_image = Image.fromarray(image_array.astype('uint8'))
    input = pil_image.convert("RGBA")
    input = remove(input)

  random_bg = random.randint(1,11)

  background_image_path = f"./backgrounds/bg{random_bg}.jpg"
  bg = Image.open(background_image_path).convert("RGBA")
  bg = bg.resize(input.size, Image.ANTIALIAS)

  composite_image = Image.alpha_composite(bg, input)
  composite_image = composite_image.convert("RGB")
  composite_image_array = tf.keras.utils.img_to_array(composite_image)

  img = tf.constant(np.array(composite_image_array))
  base_shape = tf.shape(img)[:-1]
  float_base_shape = tf.cast(base_shape, tf.float32)

  for n in range(0, 1):
    new_shape = tf.cast(float_base_shape*(octave_scale**n), tf.int32)

    img = tf.image.resize(img, new_shape).numpy()

    img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)

  img = tf.image.resize(img, base_shape)
  img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)

  crazy_gen_resized = tf.image.resize(img, (400, 400))

  return np.array(crazy_gen_resized/255.)

button= st.button("click to generate image")

if button :
    image = go_crazy()
    st.write(image)
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.axis("off")
    st.pyplot(fig)
