// import tf from "@tensorflow/tfjs-node"
import {ITensorflow, LayersModel, SymbolicTensor, Tensor} from "./tensorflowLoader"

export const imageDim = [28, 28, 1] as const
/** Width of the encoder output */
export const latentDim = 16

type ILayers = ITensorflow["layers"]
type ILayer = ILayers["Layer"]
type ILayerData = ILayer

// const tfTyping: ITensorflow = <any>null

const applyIt = (x: ILayer, y: ILayer): ILayer => (y as any).apply(x as any) as any

const chainSequentialLayers = (layers: ILayerData[], seed?: ILayerData) => (seed ? layers.reduce(applyIt, seed) : layers.reduce(applyIt))

const wrapInModel = (inputs: SymbolicTensor, outputs: ILayer, tf: ITensorflow) => tf.model({inputs, outputs: <any>outputs})

export const setupEncoder = (tf: ITensorflow) => {
	const encoderLayers: ILayerData[] = [
		<any>tf.input({shape: <any>imageDim, name: "encoder_input"}),
		tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 1, activation: "relu"}),
		tf.layers.maxPool2d({poolSize: 2}),
		tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 1, activation: "relu"}),
		// tf.layers.maxPool2d({poolSize: 2}),  Is this any use?  Need maxPool before flatten?
		tf.layers.flatten({}),
		tf.layers.dense({units: latentDim, activation: "relu", name: "encoder_output"}),
	]

	const encoder = wrapInModel(encoderLayers[0] as any, chainSequentialLayers(encoderLayers), tf)
	return encoder
}

export const setupDecoder = (tf: ITensorflow) => {
	const decoderLayers: ILayerData[] = [
		<any>tf.input({shape: [latentDim], name: "decoder_input"}),
		tf.layers.dense({units: 11 * 11 * 16}),
		tf.layers.reshape({targetShape: [11, 11, 16]}),
		tf.layers.conv2dTranspose({filters: 16, kernelSize: 3}), // Output: 13x13
		tf.layers.upSampling2d({}), // Output: 26x26  - We don't have to do this, could just widen the convolution with wide filters
		tf.layers.conv2dTranspose({filters: 1, kernelSize: 3}), // Output: 28x28
	]
	const decoder = wrapInModel(decoderLayers[0] as any, chainSequentialLayers(decoderLayers), tf)

	return decoder
}

export const setupAutoEncoder = (encoder: LayersModel, decoder: LayersModel, tf: ITensorflow) => {
	const inputs = encoder.inputs
	// const encoderOutputs = <any[]>encoder.apply(inputs)  // What does this mean?
	const encoderOutput = encoder.apply(inputs) // ? Is this a hack to get the outputs?  The last layer of the encoder contains the outputs.  Here: there is ATW just a single output
	const encoded = encoderOutput // encoderOutputs[2]
	const decoderOutput = decoder.apply(<any>encoded)
	const v = tf.model({
		inputs: inputs,
		outputs: [<any>decoderOutput, <any>encoderOutput], // Both the final decoder output, and the encoder's output - the latent space - is outputs of the model.
		name: "autoEncoderModel",
	})

	v.summary()
	return v
}

/**
 * The custom loss function for auto encoder.  Soon to be extended to vae.
 *
 * @param {tf.tensor} inputs the encoder inputs a batched image tensor
 * @param {tf.tensor} output the encoder and decoder outputs
 */
export function autoEncoderLoss(inputs: Tensor, outputs: Tensor[], tf: ITensorflow) {
	return tf.tidy(() => {
		const originalDim = inputs.shape[1] ?? -1 // NB:  Should probably be the product of dimensions
		const decoderOutput = outputs[0] // outputs[1] is the latent vector
		/*	  No VAE just yet
const zMean = outputs[1];
	  const zLogVar = outputs[2]; */

		// First we compute a 'reconstruction loss' terms. The goal of minimizing
		// this term is to make the model outputs match the input data.
		const reconstructionLoss = tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim)

		// binaryCrossEntropy can be used as an alternative loss function
		// const reconstructionLoss =
		//  tf.metrics.binaryCrossentropy(inputs, decoderOutput).mul(originalDim);

		// Next we compute the KL-divergence between zLogVar and zMean, minimizing
		// this term aims to make the distribution of latent variable more normally
		// distributed around the center of the latent space.
		/*	  
		No klLoss so far
		let klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp());
	  klLoss = klLoss.sum(-1).mul(-0.5); */

		return reconstructionLoss.mean() // .add(klLoss).mean();
	})
}
