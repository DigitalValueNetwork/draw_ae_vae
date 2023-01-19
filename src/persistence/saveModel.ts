import tf from "@tensorflow/tfjs-node"
import path from "path"
import mkdirp from "mkdirp"

export const saveModel = async (model: tf.LayersModel, savePath: string, title = "decoder") => {
	const decoderPath = path.join(savePath, title)
	mkdirp.sync(decoderPath)
	const saveURL = `file://${decoderPath}`
	console.log(`Saving decoder to ${saveURL}`)
	await model.save(saveURL)
}
