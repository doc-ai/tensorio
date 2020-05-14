<a name="training"></a>
## Training

* [ Introduction ](#introduction)
* [ A Basic Example ](#training-basic-example)
* [ The Batch API ](#training-batch-api)
* [ A Complete Example ](#training-complete-example)

<a name="introduction"></a>
#### Introduction

The full TensorFlow backend supports on-device training with Tensor/IO. Support for training allows you to deploy a trainable model to a phone and then train it directly on the device with local data. You use the same model.json file to describe the inputs and outputs for training and add a *train* field that identifies the training ops to run.

Training inputs will usually include both the model inputs and outputs, while the training output will be the loss value you would like to measure. The training ops will be the named operations that are responsible for executing a round of training on the model and will usually include the optimization operation.

<a name="training-basic-example"></a>
#### A Basic Example

Make sure you are using a backend which supports training and have a model with the additional ops required for training. Tell Tensor/IO that your model targets training with the *model.modes* field, and add the *train* field to your model.json:

```json
"model": {
  "file": "train",
  "backend": "tensorflow",
  "modes": ["train"]
},

"inputs": [ 
	... 
],
"outputs": [ 
	... 
],

"train": {
  "ops": [
    "training_op_name"
  ]
}
```

<a name="training-batch-api"></a>
#### The Batch API 

Unlike inference, which currently runs on a single example, training runs on many examples simultaneously and requires the use of the `TIOBatch` API. A batch is simply a collection of training examples whose key-values correspond to the named training inputs expected by the model. Think of a batch as a matrix of training data with each item occupying a row and each column a named column of values for a single input layer across every item.

Instantiate a batch with the input keys to your trainable model. This will typically include both the inputs and labels. Then add items to the batch, typed to `TIOBatchItem` but which are really just dictionaries of named values corresponding to the `TIOData` protocol:

```objc
TIOBatch *batch = [[TIOBatch alloc] initWithKeys:@[@"image", @"labels"]];

[batch addItem:@{
    @"image": cat,
    @"labels": @(0)
}];
    
[batch addItem:@{
    @"image": dog,
    @"labels": @(1)
}];
```

You can then call train on the model with this batch to execute a single round of training, equivalent to one epoch with a single batch:

```objc
NSDictionary *results = (NSDictionary *)[model train:batch];
```

As with inference, the results dictionary will contain the output of training, typically the loss function you'd like to measure.

To execute multiple epochs of training across many batches, you will need to set up an epoch loop and collect data for the batches yourself. An API to support this common practice is forthcoming.

<a name="training-complete-example"></a>
#### A Complete Example

A trainable cats vs dogs model is included with the full TensorFlow example in this repository. Inside the cats-vs-dogs-train.tiobundle you'll find the expected *model.json* file along with a *train* directory that contains the results of exporting a saved model in TensorFlow (more below).

The *model.json* looks like:

```json
{
  "name": "Cats vs Dogs MobileNet V2 1.0 128",
  "details": "Cats vs Dogs Kaggle model based on MobileNet V2 architecture with a width multiplier of 1.0 and an input resolution of 128x128.",
  "id": "cats-vs-dogs-v2-100-128-unquantized",
  "version": "1",
  "author": "doc.ai",
  "license": "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0",
  "model": {
    "file": "train",
    "quantized": false,
    "type": "image.classification.catsvsdogs",
    "backend": "tensorflow",
    "modes": ["train"]
  },
  "inputs": [
    {
      "name": "image",
      "type": "image",
      "shape": [-1,128,128,3],
      "format": "RGB",
      "normalize": { "standard": "[0,1]" }
    },
    {
      "name": "labels",
      "type": "array",
      "dtype": "int32",
      "shape": [-1,1]
    }
  ],
  "outputs": [
    {
      "name": "sigmoid_cross_entropy_loss/value",
      "type": "array",
      "shape": [1]
    }
  ],
  "train": {
    "ops": [
      "train"
    ]
  }
}
```

Notice especially the addition of the *train* field with its *ops* parameter and that the shape of the two inputs includes a batch dimension, identified in TensorFlow by a `-1` along the first axis. The names of the inputs and of the training op have been taken from the graph, snippets of which are included below. The name of the output comes from an inspection of the exported graph using TensorFlow's *saved_model_cli*, also below.

Train this model with the `TIOBatch` API:

```objc
TIOBatch *batch = [[TIOBatch alloc] initWithKeys:@[@"image", @"labels"]];
    
[batch addItem:@{
    @"image": cat,
    @"labels": @(0)
}];
    
[batch addItem:@{
    @"image": dog,
    @"labels": @(1)
}];

for (NSUInteger epoch = 0; epoch < 100; epoch++) {
	NSDictionary *results = (NSDictionary *)[model train:batch];
	NSLog(@"%@", results[@"sigmoid_cross_entropy_loss/value"]);
}
```

**Model Snippets**

For a complete set of examples showing how to build models for training on device with Tensor/IO, see the [TensorIO Example](https://github.com/doc-ai/tensorio-examples) repository.

This model was exported from the following code. Notice that the `serving_input_receive_fn` provides an input named *image*, that we are exporting the model using `experimental_mode=tf.estimator.ModeKeys.TRAIN`, and that in the `model_fn` we set up a placeholder for the `labels` and name it *labels* and name the training op *train*. The names in the *model.json* file correspond directly to these values.

This model was built with TensorFlow 1.13. 

```python
# trainable model snippets

# service_input_receive_fn used by estimator.export_saved_model

def serving_input_receiver_fn(params):
  dimension = [None, params['target_dim'], params['target_dim'], 3]

  inputs = {
    'image': tf.placeholder(tf.float32, dimension, name='image'),
  }

  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# the save_model function which is called by a custom python script
# you must have already trained the model for at least a single epoch and generated training checkpoints
# the model_dir param points to that checkpoints directory

def save_model(model_dir, output_dir, dims):
  input_params = {'target_dim': dims}
  estimator = tf.estimator.Estimator(
    model_fn=model.model_fn, 
    model_dir=model_dir)
  estimator.export_saved_model(
    output_dir, 
    lambda:serving_input_receiver_fn(input_params),
    as_text=False,
    experimental_mode=tf.estimator.ModeKeys.TRAIN)
    
# the model_fn expected by tensorflow's estimator api
# note: the labels placeholder if labels is None
# note: the named optimization op
 
def model_fn(features, labels, mode, params):
  
  MOBILENET = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2'

  # build model layers

  module = hub.Module(MOBILENET)
  feature_vector = module(features["image"])

  logits = tf.layers.dense(feature_vector, 1, name='logit')
  probabilities = tf.nn.sigmoid(logits, name='sigmoid')

  # prepare predictions

  predictions = {
    'probability': probabilities,
    'class': tf.to_int32(probabilities > 0.5)
  }
  prediction_output = tf.estimator.export.PredictOutput({
    'probability': probabilities,
    'class': tf.to_int32(probabilities > 0.5)
  })

  # return an estimator spec for prediction before computing a loss

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode, 
      predictions=predictions,
      export_outputs={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output
      })

  # calculate loss

  if labels is None: # during training export
    labels = tf.placeholder(tf.int32, shape=(1), name='labels')

  labels = tf.reshape(labels, [-1,1])
  labels = tf.cast(labels, tf.float32)

  loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=labels,
    logits=logits
  )

  # calculate accuracy metric

  accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["class"], name='accuracy')

  if mode == tf.estimator.ModeKeys.TRAIN:

    # generate some summary info
    # these ops are not supported by the TensorFlow mobile build

    # unsupported ops on mobile build
    # tf.summary.scalar('average_loss', loss)
    # tf.summary.scalar('accuracy', accuracy[1])

    # prepare an optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step(),
      name="train")

    # return an estimator spec

    return tf.estimator.EstimatorSpec(
      mode=mode, 
      loss=loss, 
      train_op=train_op)
  
  if mode == tf.estimator.ModeKeys.EVAL:

    # add evaluation metrics
    
    eval_metric_ops = {
      "accuracy": accuracy
    }

    # return an estimator spec

    return tf.estimator.EstimatorSpec(
      mode=mode, 
      loss=loss, 
      eval_metric_ops=eval_metric_ops)
```

We can use tensorflow's *saved_model_cli* to give us the inputs and outputs to this model. We already know the input is named "image" and we learn that the output corresponds to the sigmoid cross entropy loss, which we use for our model outputs field:

```bash
$ saved_model_cli show --dir {export-dir} --all

MetaGraphDef with tag-set: 'train' contains the following SignatureDefs:

signature_def['train']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['image'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 128, 128, 3)
        name: image:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['loss'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: sigmoid_cross_entropy_loss/value:0
  Method name is: tensorflow/supervised/training
```

Find more examples of models that have been exported for training and how to use them at [tensorio/examples](https://github.com/doc-ai/tensorio/tree/master/examples).

<< [Image Data](ImageData.md) || [TOC](TOC.md) || [Example Models](ExampleModels.md) >>