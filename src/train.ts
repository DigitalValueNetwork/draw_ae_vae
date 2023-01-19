import tf from "@tensorflow/tfjs-node"
import {concatMap, from, map, Observable} from "rxjs"
import {generateImage} from "./generate/imageGenerator.js"
import {autoEncoderLoss, imageDim, latentDim, setupAutoEncoder, setupDecoder, setupEncoder} from "./model.js"

export const train = async (chunks: Observable<{buffer: number[]; length: number}>, imagePreview: (imageTensor: tf.Tensor) => Promise<void>) => {
	const encoder = setupEncoder()
	const decoder = setupDecoder()
	const autoEncoder = setupAutoEncoder(encoder, decoder)

	const optimizer = tf.train.adam()

	await chunks
		.pipe(
			map(({buffer, length}, idx) => ({tensor: tf.tensor4d(buffer, [length, ...imageDim]), idx})),
			concatMap(({tensor, idx}) => {
				let epochAction = null
				optimizer.minimize(() => {
					const outputs = autoEncoder.apply(tensor)
					const loss = autoEncoderLoss(tensor, <any>outputs)

					process.stdout.write(".")
					if (idx % 50 === 0) {
						console.log("\nLoss:", loss.dataSync()[0])
					}
					if (idx % 100 === 0 && idx > 0) {
						epochAction = (async () => {
							console.log("\n" + `Epoch: ${idx / 100}`)
							await generateImage(decoder, (<tf.Tensor[]>outputs)[1], async imageTensor => await imagePreview(Array.isArray(imageTensor) ? imageTensor[0] : imageTensor)).catch(err => {
								console.log(err)
							})
						})()
					} else epochAction = null

					return <tf.Scalar>loss
				})
				tf.dispose([tensor])
				return epochAction ? from(epochAction) : from([])
			})
		)
		.forEach(() => {})
	// saveModel(decoderModel, );
}
