import {LayersModel} from "@tensorflow/tfjs-node"
import path from "path"
import mkdirp from "mkdirp"

export const saveModel = async (model: LayersModel, savePath: string, title = "decoder") => {
	const modelPath = path.join(savePath, title)
	mkdirp.sync(modelPath)
	const saveURL = `file://${modelPath}`
	console.log(`Saving ${title} to ${saveURL}`)
	await model.save(saveURL)
}
