// import tf from "@tensorflow/tfjs-node"
import { ITensorflow, LayersModel, Tensor } from "../tensorflowLoader"

/**
 * Generate an image
 *
 * @param {LayersModel} decoderModel Decoder portion of the VAE.
 * @param {number | Tensor} latentDimSizeOrTensor A latent space tensor, or dimensionality of the latent space for generating a sample.
 */
export const generateImage = async (decoderModel: LayersModel, latentDimSizeOrTensor: number | Tensor , cb: (image: Tensor | Tensor[]) => Promise<void>, tf: ITensorflow) => {
	const targetZ = typeof latentDimSizeOrTensor === "number" ? tf.zeros([latentDimSizeOrTensor]).expandDims() : latentDimSizeOrTensor
	const generated = decoderModel.predict(targetZ)

	await cb(generated)
	tf.dispose([targetZ, generated])
}
