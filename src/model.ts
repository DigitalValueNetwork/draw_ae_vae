// import tf from "@tensorflow/tfjs-node"
import {ITensorflow, LayersModel, SymbolicTensor, Tensor} from "./tensorflowLoader"
import {createZLayerClass} from "./ZLayer.js"

// export const imageDim = [28, 28, 1] as const
/** Width of the encoder output */
export const latentDim = 3

type ILayers = ITensorflow["layers"]
type ILayer = ILayers["Layer"]
type ILayerData = ILayer

// const tfTyping: ITensorflow = <any>null

const applyIt = (x: ILayer, y: ILayer | ILayer[]): ILayer =>
	[
		[...(Array.isArray(y) ? (y as any) : [y])].map(y => {
			const l = y.apply(x as any) as any
			return l
		}),
	].map(x => (x.length === 1 ? x[0] : x))[0]

const chainSequentialLayers = (layers: (ILayerData | ILayerData[])[], seed?: ILayerData) => (seed ? layers.reduce(applyIt, seed) : layers.reduce(applyIt as any))

const wrapInModel = (inputs: SymbolicTensor, outputs: ILayer | ILayer[], name: string, tf: ITensorflow) => tf.model({inputs, outputs: <any>outputs, name})

export const setupEncoder = (tf: ITensorflow, imageDim: readonly [number, number, number]) => {
	const ZLayer = createZLayerClass(tf)

	const encoderLayers: ILayerData[] = [
		<any>tf.input({shape: <any>imageDim, name: "encoder_input"}),
		tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 1, activation: "relu"}),
		tf.layers.maxPool2d({poolSize: 2}),
		tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, activation: "relu"}),
		// tf.layers.maxPool2d({poolSize: 2}),  Is this any use?  Need maxPool before flatten?
		tf.layers.flatten({}),
		[tf.layers.dense({units: latentDim, /* activation: "relu", */ name: "z_mean"}), tf.layers.dense({units: latentDim, /* activation: "relu", */ name: "z_log_var"})],
		// new ZLayer({name: "z-layer", outputShape: [latentDim]})
	]

	const [zMean, zLogVar] = chainSequentialLayers(encoderLayers) as ILayer[]
	// const [zMean, zLogVar] = (sequence as any).sourceLayer.inboundNodes[0].inboundLayers[0]
	const z = chainSequentialLayers([[zMean, zLogVar], new ZLayer({name: "z-layer", outputShape: [latentDim]})] as any[])

	const encoder = wrapInModel(encoderLayers[0] as any, [zMean, zLogVar, z as any], "encoder", tf)
	return encoder
}

export const setupDecoder = (tf: ITensorflow) => {
	const decoderLayers: ILayerData[] = [
		// Todo: create one of 28x28x1 and one for 128x128x3
		<any>tf.input({shape: [latentDim], name: "decoder_input"}),
		tf.layers.dense({units: 61 * 61 * 13, activation: "relu"}),
		tf.layers.reshape({targetShape: [61, 61, 13]}),
		tf.layers.conv2dTranspose({filters: 13, kernelSize: 3, activation: "relu"}), // Output: 63x63
		tf.layers.upSampling2d({}), // Output: 126x126  - We don't have to do this, could just widen the convolution with wide filters
		tf.layers.conv2dTranspose({filters: 3, kernelSize: 3}), // Output: 128x128x3
	]
	const decoder = wrapInModel(decoderLayers[0] as any, chainSequentialLayers(decoderLayers), "decoder", tf)

	return decoder
}

export const setupAutoEncoder = (encoder: LayersModel, decoder: LayersModel, tf: ITensorflow) => {
	const inputs = encoder.inputs
	// const encoderOutputs = <any[]>encoder.apply(inputs)  // What does this mean?
	const encoderOutputs = encoder.apply(inputs) as any[] // ? Is this a hack to get the outputs?  The last layer of the encoder contains the outputs.  Here: there is ATW just a single output
	const encoded = encoderOutputs[2]
	const decoderOutput = decoder.apply(<any>encoded)
	const v = tf.model({
		inputs: inputs,
		outputs: [<any>decoderOutput, ...encoderOutputs], // Both the final decoder output, and the encoder's output - the latent space - is outputs of the model.
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
		const originalDim = inputs.shape[1] ?? -1
		// Outputs: [decoderOutput, zMean, zLogDev, latent] - see above
		const decoderOutput = outputs[0] // outputs[1] is the latent vector
		const zMean = outputs[1]
		const zLogVar = outputs[2]

		// First we compute a 'reconstruction loss' terms. The goal of minimizing
		// this term is to make the model outputs match the input data.
		const reconstructionLoss = tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim * originalDim)

		// binaryCrossEntropy can be used as an alternative loss function
		// const reconstructionLoss =
		//  tf.metrics.binaryCrossEntropy(inputs, decoderOutput).mul(originalDim);

		// Next we compute the KL-divergence between zLogVar and zMean, minimizing
		// this term aims to make the distribution of latent variable more normally
		// distributed around the center of the latent space.
		let klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp()).sum(-1).mul(-0.5)

		return reconstructionLoss.add(klLoss).mean()
	})
}
