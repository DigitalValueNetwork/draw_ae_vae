import tf from "@tensorflow/tfjs-node"
import {map, Observable} from "rxjs"
import {autoEncoderLoss, imageDim, setupAutoEncoder, setupDecoder, setupEncoder} from "./model.js"

export const train = async (chunks: Observable<{buffer: number[]; length: number}>) => {
	const encoder = setupEncoder()
	const decoder = setupDecoder()
	const autoEncoder = setupAutoEncoder(encoder, decoder)

	const optimizer = tf.train.adam()

	await chunks.pipe(map(({buffer, length}, idx) => ({tensor: tf.tensor4d(buffer, [length, ...imageDim]), idx}))).forEach(({tensor, idx}) => {
		optimizer.minimize(() => {
			const outputs = autoEncoder.apply(tensor)
			const loss = autoEncoderLoss(tensor, <any>outputs)

			process.stdout.write(".")
			if (idx % 50 === 0) {
				console.log("\nLoss:", loss.dataSync()[0])
			}

			return <tf.Scalar>loss
		})
		tf.dispose([tensor])
	})
}
