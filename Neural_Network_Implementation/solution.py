from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - parameter_grad * self.lr
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = updater.inertia * self.momentum + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.where(inputs >= 0, inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.where(self.forward_inputs >= 0, grad_outputs, 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        exp_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        return (grad_outputs - (self.forward_outputs * grad_outputs).sum(keepdims=True, axis=1)) * self.forward_outputs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return np.dot(inputs, self.weights.T) + self.biases.T
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.biases_grad = grad_outputs.sum(axis=0)
        self.weights_grad = np.dot(grad_outputs.T, self.forward_inputs)
        return np.dot(grad_outputs, self.weights)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        return np.array([-np.mean(y_gt * np.log(y_pred_clipped))])
    
    def gradient_impl(self, y_gt, y_pred):
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        return -y_gt / y_pred_clipped / y_gt.shape[0]

# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def forward_impl(self, inputs):
        n, d, h, w = inputs.shape

        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            var = np.var(inputs, axis=(0, 2, 3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()

            self.forward_centered_inputs = inputs - mean
            self.forward_inverse_std = 1 / np.sqrt(var + eps)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std
        else:
            self.forward_centered_inputs = inputs - self.running_mean.reshape(1, d, 1, 1)
            self.forward_inverse_std = 1 / np.sqrt(self.running_var.reshape(1, d, 1, 1) + eps)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std

        return self.gamma.reshape(1, d, 1, 1) * self.forward_normalized_inputs + self.beta.reshape(1, d, 1, 1)

    def backward_impl(self, grad_outputs):
        n, d, h, w = grad_outputs.shape

        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        grad_norm = grad_outputs * self.gamma.reshape(1, d, 1, 1)
        grad_var = np.sum(grad_norm * self.forward_centered_inputs, axis=(0, 2, 3), keepdims=True) * (-0.5) * self.forward_inverse_std**3
        grad_mean = np.sum(grad_norm, axis=(0, 2, 3), keepdims=True) * (-self.forward_inverse_std)

        grad_input = (grad_norm * self.forward_inverse_std + 
                     grad_var * 2 * self.forward_centered_inputs / (n * h * w) + 
                     grad_mean / (n * h * w))

        return grad_input

# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def forward_impl(self, inputs):
        n, d, ih, iw = inputs.shape
        p = self.pool_size
        oh, ow = ih // p, iw // p

        x_reshaped = inputs.reshape(n, d, oh, p, ow, p)
        x_blocks = x_reshaped.transpose(0, 1, 2, 4, 3, 5)
        x_blocks_flat = x_blocks.reshape(n, d, oh, ow, -1)

        if self.pool_mode == "max":
            self.forward_idxs = x_blocks_flat.argmax(axis=-1)
            return x_blocks_flat.max(axis=-1)
        else:
            return x_blocks_flat.mean(axis=-1)

    def backward_impl(self, grad_outputs):
        if self.pool_mode == "max":
            n, d, oh, ow = grad_outputs.shape
            p = self.pool_size
            ih, iw = oh * p, ow * p
            gradient_inputs = np.zeros((n, d, ih, iw), dtype=grad_outputs.dtype)

            oh_indices, ow_indices = np.indices((oh, ow))
            h_start = oh_indices * p
            w_start = ow_indices * p

            row = self.forward_idxs // p
            col = self.forward_idxs % p

            h_coords = h_start[None, None, :, :] + row
            w_coords = w_start[None, None, :, :] + col

            n_idx = np.arange(n)[:, None, None, None]
            d_idx = np.arange(d)[None, :, None, None]
            gradient_inputs[n_idx, d_idx, h_coords, w_coords] += grad_outputs

            return gradient_inputs
        else:
            p = self.pool_size
            return np.repeat(np.repeat(grad_outputs, p, axis=2), p, axis=3) / (p**2)

# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def forward_impl(self, inputs):
        if self.is_training:
            self.forward_mask = (np.random.uniform(size=inputs.shape) > self.p).astype(inputs.dtype)
            return inputs * self.forward_mask
        else:
            return inputs * (1 - self.p)

# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=0.01, momentum=0.9)
    model = Model(loss, optimizer)

    model.add(Conv2D(32, kernel_size=3, input_shape=(3, 32, 32)))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode="max"))

    model.add(Conv2D(64, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode="max"))

    model.add(Conv2D(128, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode="max"))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    model.fit(x_train, y_train, batch_size=256, epochs=6, 
              x_valid=x_valid, y_valid=y_valid, shuffle=True)

    return model
