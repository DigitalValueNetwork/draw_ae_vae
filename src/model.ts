// import tf from "@tensorflow/tfjs-node"
import {ITensorflow, LayersModel, SymbolicTensor, Tensor} from "./tensorflowLoader"
import {createZLayerClass} from "./ZLayer.js"

// export const imageDim = [28, 28, 1] as const
/** Width of the encoder output */
export const latentDim = 5

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

	// Switch to the ChatGPT suggestion - much faster to train

	const encoderLayers: ILayerData[] = [
		<any>tf.input({shape: <any>imageDim, name: "encoder_input"}),
		tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 1, activation: "relu"}),
		tf.layers.maxPooling2d({poolSize: 2}),
		tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 1, activation: "relu"}),
		tf.layers.maxPooling2d({poolSize: 2}),
		tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, activation: "relu"}),
		tf.layers.maxPooling2d({poolSize: 2}), //  Is this any use?  Need maxPool before flatten?    By having a last pooling, the number of weights to the latent dim is 4 times smaller. This de-smartifies the network, but perhaps not with much
		tf.layers.flatten({}),
		[tf.layers.dense({units: latentDim, /* activation: "relu", */ name: "z_mean"}), tf.layers.dense({units: latentDim, /* activation: "relu", */ name: "z_log_var"})],
	]

	const [zMean, zLogVar] = chainSequentialLayers(encoderLayers) as ILayer[]
	const z = chainSequentialLayers([[zMean, zLogVar], new ZLayer({name: "z-layer", outputShape: [latentDim]})] as any[])

	const encoder = wrapInModel(encoderLayers[0] as any, [zMean, zLogVar, z as any], "encoder", tf)
	// encoder.summary()
	return encoder
}

export const setupDecoder = (tf: ITensorflow) => {
	const decoderLayers: ILayerData[] = [
		// Todo: create one of 28x28x1 and one for 150x200
		// https://madebyollin.github.io/convnet-calculator/
		<any>tf.input({shape: [latentDim], name: "decoder_input"}),
		tf.layers.dense({units: 33 * 46 * 64, activation: "relu"}), // To big, switch to chatGPT solution  (37,50,32) (No, don't think this works, but reduce the params here to reduce model size, a lot)
		// Set this up, so that the first conv is low resolution with lots of filters, reading from the latent dim - then add resolution
		// Should merge back into loadImage branch.
		tf.layers.reshape({targetShape: [33, 46, 64]}),
		tf.layers.conv2dTranspose({filters: 64, kernelSize: 3, activation: "relu"}), // Output: 35x48
		tf.layers.upSampling2d({}), // Output: 70x96
		tf.layers.conv2dTranspose({filters: 64, kernelSize: 3, activation: "relu"}), // Output: 72x98
		tf.layers.upSampling2d({}), // Output: 144x196
		tf.layers.conv2dTranspose({filters: 64, kernelSize: 3, activation: "relu"}), // Output: 146x198
		tf.layers.conv2dTranspose({filters: 9, kernelSize: 3, activation: "relu"}), // Output: 148x200
		tf.layers.conv2dTranspose({filters: 3, kernelSize: 3}), // Output: 150x202x3
		tf.layers.cropping2D({
			cropping: [
				[0, 0],
				[1, 1],
			],
		}), // Adjust the y axis down again
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
		const zMean = outputs[1] // shape: [batch, 3]
		const zLogVar = outputs[2] // shape: [batch]

		// PROBLEM:
		//  After the first iteration, all the output values turns NaN.
		//      * What is the the value of klLoss ?
		//          * klLoss sometimes hold very high values, from very high random values.  What is the values during mnist?
		//      * What happens in the ZLayer - any adjustments needed for more channels?
		//      * Why is mean squaredError so high?  Should it not be a mean? Re-iterate on this concept and check the values again. Maybe the output's are forced out of alignment.
		//         * When trying again - the reconstructionLoss vas 24000 - which makes perfect sense - with random outputs around 0, and inputs around 128-255
		// There might be an issue in the zMean and zLogVar - and further investigation into how these works - might be needed.
		// It's also a mystery - that the values of the output are forced to NaN - what gradients are doing that.

		// First we compute a 'reconstruction loss' terms. The goal of minimizing
		// this term is to make the model outputs match the input data.
		const reconstructionLoss = tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim)

		// binaryCrossEntropy can be used as an alternative loss function
		// const reconstructionLoss =
		//  tf.metrics.binaryCrossEntropy(inputs, decoderOutput).mul(originalDim);

		// Next we compute the KL-divergence between zLogVar and zMean, minimizing
		// this term aims to make the distribution of latent variable more normally
		// distributed around the center of the latent space.
		const klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp()).sum(-1).mul(-0.5)
		return reconstructionLoss.add(klLoss).mean()
	})
}
