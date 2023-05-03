import {Scalar} from "@tensorflow/tfjs-node"
import {range} from "radash"
import {filter, map, Observable, lastValueFrom} from "rxjs"
import {generateImage} from "./generate/imageGenerator.js"
import {autoEncoderLoss, setupAutoEncoder, setupDecoder, setupEncoder} from "./model.js"
import {ITensorflow, Tensor} from "./tensorflowLoader.js"

export const train = async (chunks: Observable<{ buffer: number[]; length: number }>, epochCount: number, imagePreview: (imageTensor: any) => Promise<void>, tf: ITensorflow) => {
	const imageDim = [150, 200, 3] as const
	const encoder = setupEncoder(tf, imageDim)
	const decoder = setupDecoder(tf)
	const autoEncoder = setupAutoEncoder(encoder, decoder, tf)

	const optimizer = tf.train.adam()

	for (const {epoch, last} of [...range(1, epochCount)].map(epoch => ({epoch, last: epoch === epochCount}))) {
		console.log("\n" + `Epoch: ${epoch} of ${epochCount}`)
		const lastLatentSpaceVectors = await lastValueFrom(
			chunks.pipe(
				filter(() => 5 / 6 < Math.random()),
				map(({buffer, length}, idx) => ({tensor: tf.tensor4d(buffer, [length, ...imageDim]), idx})),
				map(({tensor, idx}) => {
					let latentSpaceVectors: number[][] = <any>null // Trying to store a tensor here will fail, it will be disposed
					optimizer.minimize(() => {
						const outputs = autoEncoder.apply(tensor)
						const loss = autoEncoderLoss(tensor, <any>outputs, tf)

						process.stdout.write(".")
						if (idx % 50 === 0) {
							console.log("\nLoss:", loss.dataSync()[0])
						}
						latentSpaceVectors = (<any>(<Tensor[]>outputs)[1]).arraySync()
						return <Scalar>loss
					})
					tf.dispose([tensor])
					return latentSpaceVectors // epochAction ? from(epochAction) : from([])
				})
			)
		)

		console.log("\n")
		await generateImage(decoder, tf.tensor([lastLatentSpaceVectors[0]]), async imageTensor => await imagePreview(Array.isArray(imageTensor) ? imageTensor[0] : imageTensor), tf).catch(err => {
			console.log(err)
		})
		if (last) {
			console.log(JSON.stringify(lastLatentSpaceVectors))
		}
	}

	return decoder
}
