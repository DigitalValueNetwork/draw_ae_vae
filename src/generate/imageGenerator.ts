import tf from "@tensorflow/tfjs-node"

/**
 * Generate an image
 *
 * @param {tf.LayersModel} decoderModel Decoder portion of the VAE.
 * @param {number | tf.Tensor} latentDimSizeOrTensor A latent space tensor, or dimensionality of the latent space for generating a sample.
 */
export const generateImage = async (decoderModel: tf.LayersModel, latentDimSizeOrTensor: number | tf.Tensor , cb: (image: tf.Tensor | tf.Tensor[]) => Promise<void>) => {
	const targetZ = typeof latentDimSizeOrTensor === "number" ? tf.zeros([latentDimSizeOrTensor]).expandDims() : latentDimSizeOrTensor
	const generated = decoderModel.predict(targetZ)

	await cb(generated)
	tf.dispose([targetZ, generated])
}
