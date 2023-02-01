import {ITensorflow, Shape, Tensor} from "./tensorflowLoader.js"

// type IX = ITensorflow["layers"]["Layer"]

export const createZLayerClass = (tf: ITensorflow) => {
	const { Layer } = tf.layers

	/**
	 * This layer implements the 'reparameterization trick' described in
	 * https://blog.keras.io/building-autoencoders-in-keras.html.
	 * Or, better here: https://www.baeldung.com/cs/vae-reparameterization
	 *
	 * The implementation is in the call method.
	 * Instead of sampling from Q(z|X):
	 *    sample epsilon = N(0,I)
	 *    z = z_mean + sqrt(var) * epsilon
	 */
	const ZL = class ZLayer extends Layer {
		constructor(config: any /* LayerArgs */) {
			super(config)
		}

		computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
			tf.util.assert(inputShape.length === 2 && Array.isArray(inputShape[0]), () => `Expected exactly 2 input shapes. But got: ${inputShape}`)
			return inputShape[0] as any
		}

		/**
		 * The actual computation performed by an instance of ZLayer.
		 *
		 * @param {Tensor[]} inputs this layer takes two input tensors, z_mean and
		 *     z_log_var
		 * @return A tensor of the same shape as z_mean and z_log_var, equal to
		 *     z_mean + sqrt(exp(z_log_var)) * epsilon, where epsilon is a random
		 *     vector that follows the unit normal distribution (N(0, I)).
		 */
		call(inputs: Tensor[]) {
			const [zMean, zLogVar] = inputs
			const batch = zMean.shape[0]
			const dim = zMean.shape[1]

			const mean = 0
			const std = 1.0
			// The reparameterization trick is about the insert of a separate parameter for the randomness, so that it does not disturb the back propagation?
			// sample epsilon = N(0, I)
			const epsilon = tf.randomNormal([batch, dim] as any, mean, std) // called three times for each optimize callback.

			// Note: multiplying the log(var) value with 0.5 before removing the log() takes the square root
			// z = z_mean + sqrt(var) * epsilon
			return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon))
		}

		static get className() {
			return "ZLayer"
		}
	}
	tf.serialization.registerClass(ZL)
	return ZL
}
