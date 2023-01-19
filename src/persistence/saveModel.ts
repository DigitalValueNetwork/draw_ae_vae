import {LayersModel} from "@tensorflow/tfjs-node"
import path from "path"

export const saveModel = async (model: LayersModel, savePath: string, title = "decoder") => {
	const decoderPath = path.join(savePath, title)
	const saveURL = `file://${decoderPath}`
	console.log(`Saving decoder to ${saveURL}`)
	await model.save(saveURL)
}
