# -*- coding: utf-8 -*-


!wget https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip
!mkdir emojis
!unzip -q openmoji-72x72-color.zip -d ./emojis


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from PIL import Image, ImageDraw
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

plt.figure(figsize=(9, 9))

for i, (j, e) in enumerate(emojis.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
    plt.xlabel(e['name'])
    plt.xticks([])
    plt.yticks([])
plt.show()

"""## Task 3: Create Examples"""

for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file

emojis

def create_example():

  image = np.ones((153, 153, 3)) * 255
  class_id = np.random.randint(0, 9)

  row = np.random.randint(0, 72)
  col = np.random.randint(0, 72)

  image[row: row+72, col: col+72, :] = np.array(emojis[class_id]['image'])

  return image.astype('uint8'), class_id, (row+10)/153, (col+10)/153

img, class_id, row, col = create_example()

plt.imshow(img)

"""## Task 4: Plot Bounding Boxes"""

def plot_bbox(image, gt_coords, pred_coords=[], norm= False):

  if norm:
    image *= 255
    image = image.astype('uint8')

  image = Image.fromarray(image)
  draw = ImageDraw.Draw(image)

  row, col = gt_coords
  row *= 153
  col *= 153 
  draw.rectangle((col, row, col+52, row+52), outline='green', width=3)

  if len(pred_coords) == 2:
    row, col = pred_coords
    row *= 153
    col *= 153 
    draw.rectangle((col, row, col+52, row+52), outline='red', width=3)


  return image

test = plot_bbox(img, [row, col])
plt.imshow(test)

"""## Task 5: Data Generator"""

def data_gen(batch_size= 16):

  while True:

    x_batch = np.zeros((batch_size, 153, 153, 3))
    y_batch = np.zeros((batch_size, 9))
    bbox_batch = np.zeros((batch_size, 2))

    for i in range(0, batch_size):

      imag, class_id, row, col = create_example()
      x_batch[i] = imag / 255
      y_batch[i, class_id] = 1
      bbox_batch[i] = np.array([row, col])

    ytrain = {'class_out': y_batch, 'box_out': bbox_batch}

    yield (x_batch, ytrain)

example, label = next(data_gen(1))

example = np.squeeze(example)
class_id = np.argmax(np.squeeze(label['class_out']))
coords = np.squeeze(label['box_out'])

example = plot_bbox(example, coords, norm= True)
plt.imshow(example)
plt.title(emojis[class_id]['name'])
plt.show()

"""## Task 6: Model"""

input_ = Input((153, 153, 3), name = 'image')

x = input_

for i in range(0, 5):
  nb_filters = 2**(4 + i)
  x = Conv2D(nb_filters, 3, activation='relu')(x)
  x = BatchNormalization()(x)
  x = MaxPool2D(2)(x)

x = Flatten()(x)

x = Dense(256, activation= 'relu')(x)

class_out = Dense(9, activation='softmax', name='class_out')(x)
box_out = Dense(2, name='box_out')(x)


model = tf.keras.models.Model(input_, [class_out, box_out])
model.summary()

"""## Task 7: Custom Metric: IoU"""

class IOU(keras.metrics.Metric):

  def __init__(self, **kwargs):
    super(IOU, self).__init__(**kwargs)

    self.iou = self.add_weight(name='iou', initializer='zeros')
    self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
    self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

  def update_state(self, y_pred, y_true, sample_weight=None):
    def get_box(y):
      rows, cols = y[:, 0], y[:, 1]
      rows, cols = rows*153, cols*153
      y1, y2 = rows, rows + 52
      x1, x2 = cols, cols + 52

      return x1, y1, x2, y2

    def get_area(x1, y1, x2, y2):
      return tf.math.abs(x2-x1) * tf.math.abs(y2-y1)

    gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
    p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

    i_x1 = tf.maximum(gt_x1, p_x1)
    i_y1 = tf.maximum(gt_y1, p_y1)
    i_x2 = tf.maximum(gt_x2, p_x2)
    i_y2 = tf.maximum(gt_y2, p_y2)

    i_area = get_area(i_x1, i_y1, i_x2, i_y2)
    u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area

    iou = i_area / u_area
    self.num_ex.assign_add(1)
    self.total_iou.assign_add(tf.reduce_mean(iou))
    self.iou = self.total_iou / self.num_ex


  def result(self):
      return self.iou

  def reset_state(self):
      self.iou = self.add_weight(name='iou', initializer='zeros')
      self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
      self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

"""## Task 8: Compile the Model"""

model.compile(
    loss={
        'class_out': 'categorical_crossentropy',
        'box_out': 'mse'
    },
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics = {
        'class_out': 'accuracy',
        'box_out': IOU(name='iou')
    }
)

"""## Task 9: Custom Callback: Model Testing"""

def test_model(model, test_gen):
  data, labels = next(test_gen)

  x = data
  y = np.squeeze(labels['class_out'])
  gt_coords = np.squeeze(labels['box_out'])

  pred_y, pred_box = model.predict(x)

  pred_coords = np.squeeze(pred_box)
  pred_class = np.argmax(np.squeeze(pred_y))

  

  gt = emojis[np.argmax(y)]['name']
  pred = emojis[pred_class]['name']

  testImg = plot_bbox(np.squeeze(data), gt_coords, pred_coords, norm= True)
  color = 'green' if gt == pred else 'red'

  plt.imshow(testImg)
  plt.xlabel(f'Pred : {pred}', color= color)
  plt.ylabel(f'GT: {gt}', color = color)
  plt.xticks([])
  plt.yticks([])

def test(model):
  plt.figure(figsize=(16, 4))

  for i in range(0, 6):
    plt.subplot(1, 6, i + 1)
    test_model(model, data_gen(1))

  plt.show()

test(model)

class ShowTestImages(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
      return test(self.model)

"""## Task 10: Model Training"""

def lr_schedule(epoch, lr):
  if (epoch + 1 ) % 5 == 0:
    lr *= 0.2
  return tf.maximum(lr, 3e-7)

model.fit(
    data_gen(), 
    epochs=50, 
    steps_per_epoch= 600,
    callbacks = [
        ShowTestImages(),
        # keras.callbacks.EarlyStopping(monitor="box_out_iou", patience=5, mode='max'),
        keras.callbacks.LearningRateScheduler(lr_schedule)
    ])
