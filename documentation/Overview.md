<a name="overview"></a>
## Overview

Tensor/IO supports many kinds of models with multiple input and output layers of different shapes and kinds but with minimal boilerplate code. In fact, you can run a variety of models without needing to write any model specific code at all.

Instead, Tensor/IO relies on a JSON description of the model that you provide. During inference, the library matches incoming data to the model layers that expect it, performing any transformations that are needed and ensuring that the underlying bytes are copied to the right place.  Once inference is complete, the library copies bytes from the output tensors back to native Objective-C types.

The built-in class for working with TensorFlow Lite (TF Lite) models, `TIOTFLiteModel`, includes support for multiple input and output layers; single-valued, vectored, matrix, and image data; pixel normalization and denormalization; and quantization and dequantization of data. In case you require a completely custom interface to a model you may specify your own class in the JSON description, and Tensor/IO will use it in place of the default class.

Although Tensor/IO supports both full TensorFlow and TF Lite models, this README will refer to TFLite throughout. Except for small differences in support of data types (`uint8_t`, `float32_t`, etc), the interface is the same.


<< [Table of Contents](TOC.md) || [Author](Author.md) >>