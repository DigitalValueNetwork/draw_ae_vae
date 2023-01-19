import tf from "@tensorflow/tfjs-node"
import {range} from "radash"
import {filter, map, Observable, lastValueFrom} from "rxjs"
import {generateImage} from "./generate/imageGenerator.js"
import {autoEncoderLoss, imageDim, setupAutoEncoder, setupDecoder, setupEncoder} from "./model.js"

export const train = async (chunks: Observable<{buffer: number[]; length: number}>, epochCount: number, imagePreview: (imageTensor: tf.Tensor) => Promise<void>) => {
	const encoder = setupEncoder()
	const decoder = setupDecoder()
	const autoEncoder = setupAutoEncoder(encoder, decoder)

	const optimizer = tf.train.adam()

	for (const epoch of range(1, epochCount)) {
		console.log("\n" + `Epoch: ${epoch} of ${epochCount}`)
		const lastOutput = await lastValueFrom(
			chunks.pipe(
				filter(() => 5 / 6 < Math.random()),
				map(({buffer, length}, idx) => ({tensor: tf.tensor4d(buffer, [length, ...imageDim]), idx})),
				map(({tensor, idx}) => {
					let lastOutput: number[][] = <any>null // Trying to store a tensor here will fail, it will be disposed
					optimizer.minimize(() => {
						const outputs = autoEncoder.apply(tensor)
						const loss = autoEncoderLoss(tensor, <any>outputs)

						process.stdout.write(".")
						if (idx % 50 === 0) {
							console.log("\nLoss:", loss.dataSync()[0])
						}
//						lastOutput = (<tf.Tensor[]>outputs)[1]
//						lastOutput = tf.tensor((lastOutput as any as tf.Tensor).arraySync())
						lastOutput = (<any>(<tf.Tensor[]>outputs)[1]).slice([0], [1]).arraySync()[0]
						return <tf.Scalar>loss
					})
					tf.dispose([tensor])
					return lastOutput  // epochAction ? from(epochAction) : from([])
				})
			)
		)

		console.log("\n")
		await generateImage(decoder, tf.tensor([lastOutput]), async imageTensor => await imagePreview(Array.isArray(imageTensor) ? imageTensor[0] : imageTensor)).catch(err => {
			console.log(err)
		})
	}

	return decoder
}
