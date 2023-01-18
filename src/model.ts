import tf from "@tensorflow/tfjs-node"

const imageDim = [28, 28, 1]

export const setupModel = () => {

	// Start with a simple convolution, from the example.  Suddenly - I can't find the example.
	// Then try with max pooling and up-sampling
	

	// Create sequentialLayers and parallelLayers function that `apply` according to name

	const encoder: tf.layers.Layer[] = [
//		tf.input({shape: [originalDim], name: 'encoder_input'})
//		tf.layers.conv2d(
	]
	const getRidOfMe = tf.layers.conv2d({
		filters: 128,
		kernelSize: 3,
		strides: 1,
		activation: 'relu',
	  })
}