# deep-belief-network
A simple, clean, fast Python implementation of Deep Belief Networks based on binary Restricted Boltzmann Machines (RBM), built upon NumPy and TensorFlow libraries in order to take advantage of GPU computation:
> Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.

> Fischer, Asja, and Christian Igel. "Training restricted Boltzmann machines: an introduction." Pattern Recognition 47.1 (2014): 25-39.

## Usage
This implementation works on Python 3. It follows [scikit-learn](http://scikit-learn.org) guidelines and in turn, can be used alongside it. Next you have a demo code for solving digits classification problem which can be found in **classification_demo.py** (check **regression_demo.py** for a regression problem and **unsupervised_demo.py** for an unsupervised feature learning problem).

Code can run either in GPU or CPU. To decide where the computations have to be performed is as easy as importing the classes from the correct module: if they are imported from _dbn.tensorflow_ computations will be carried out on GPU (or CPU depending on your hardware) using TensorFlow, if imported from _dbn_ computations will be done on CPU using NumPy. **~~Note only pre-training step is GPU accelerated so far~~ Both pre-training and fine-tuning steps are GPU accelarated**. Look the following snippet:

```python
import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
```

## Installation
I strongly recommend to use a [virtualenv](https://virtualenv.pypa.io/en/stable/) in order not to break anything of your current enviroment.

Open a terminal and type the following line, it will install the package using pip:

CPU (installs tensorflow package):
    
        pip install git+git://github.com/albertbup/deep-belief-network.git
GPU (installs tensorflow-gpu package):
    
        pip install git+git://github.com/albertbup/deep-belief-network.git@master_gpu
        
## Citing the code
BibTex reference format:

        @misc{DBNAlbert,
        title={A Python implementation of Deep Belief Networks built upon NumPy and TensorFlow with scikit-learn compatibility},
        url={https://github.com/albertbup/deep-belief-network},
        author={albertbup},
        year={2017}}

 
第1步是加载所需的库。dbn.tensorflow是一个github版本，您必须克隆存储库并将dbn文件夹粘贴到存在代码文件的文件夹中。链接到代码存储库在
https://github.com/albertbup/deep-belief-network/。

from dbn.tensorflow import SupervisedDBNClassification

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics.classification import accuracy_score
第2步是读取可以从kaggle下载的csv文件(
https://www.kaggle.com/c/digit-recognizer)。

digits = pd.read_csv("train.csv")
第3步，让我们定义我们的自变量，它们只是像素值，并以变量X的numpy数组格式存储。我们将目标变量（实际数字）存储在变量Y中。

X = np.array(digits.drop(["label"], axis=1))

Y = np.array(digits["label"])
第4步，让我们使用sklearn预处理类的方法：standardscaler。这用于转换正态分布格式的数字。

from sklearn.preprocessing import standardscaler

ss=standardscaler()

X = ss.fit_transform(X)
第5步，现在我们已经对数据进行了归一化，我们可以将它分成训练和测试集： -

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
第6步，现在我们将初始化我们的监督DBN分类器，以训练数据。

classifier = SupervisedDBNClassification(hidden_layers_structure = [256, 256],

learning_rate_rbm=0.05,

learning_rate=0.1,

n_epochs_rbm=10,

n_iter_backprop=100,

batch_size=32,

activation_function='relu',

dropout_p=0.2)
第7步，现在我们将来到训练部分，我们将使用拟合函数进行训练：

classifier.fit（X_train，Y_train）
在数据集上训练可能需要10分钟到1小时。培训完成后，我们必须检查准确性：

Y_pred = classifier.predict(X_test)

print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
