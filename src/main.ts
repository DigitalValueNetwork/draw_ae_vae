import meow from "meow"
import {loadImages, imageProps as mnistImageProps, imagesFilePath as mnistImagesFilePath} from "./mnist-loading/data.js"
import {renderImageForTerminalPreview} from "./image/terminalImage.js"
import {train} from "./train.js"
import {imageChunks, imageChunkToFlat} from "./image/imageChunks.js"
import {saveModel} from "./modelPersistence.js"

const cli = meow(
	`
Usage: 
	$ yarn start [options]

	Options
		--outputDataset Prints a CSV with the dataset
		--lookBack Number of rows to include in input
		--delay Number of rows into the future to predict
		--logDir Tensorboard output
		--saveModelPath Path to save to, if overridden to empty string - no saving should happen
		--loadModelPath Path to model to load, including file - file:///tmp/rnn_test/model.json.
`,
	{
		flags: {
			outputDataset: {
				type: "boolean",
				default: false,
			},
			outputRows: {
				type: "number",
				default: 500,
			},
			lookBack: {
				type: "number",
				default: 6,
			},
			delay: {
				type: "number",
				default: 1,
			},
			batchSize: {
				type: "number",
				default: 50,
			},
			logDir: {
				type: "string",
				default: "",
			},
			logUpdateFreq: {
				type: "string",
				default: "batch",
				optionStrings: ["batch", "epoch"],
			},
			epochs: {
				type: "number",
				default: 10,
			},
			saveModelPath: {
				type: "string",
				default: "file:///tmp/draw_ae_vae",
			},
			loadModelPath: {
				type: "string",
				default: "",
			},
		},
		importMeta: import.meta,
		allowUnknownFlags: false,
	}
)

if (cli.flags.outputDataset) {
	console.log("not implemented")
} else {
	//	const imageProps: IImgProps = { imageHeight: }

	loadImages(mnistImagesFilePath)
		.then(async images => {
			console.log(await renderImageForTerminalPreview(images[5], mnistImageProps))
			// console.log(await renderImageForTerminalPreview(images[100], mnistImageProps))
			// console.log(await renderImageForTerminalPreview(images[150], mnistImageProps))

			const model = await train(imageChunkToFlat(imageChunks(images, cli.flags.batchSize)), cli.flags.epochs, async tensor =>
				console.log(await renderImageForTerminalPreview(tensor.dataSync() as Float32Array, mnistImageProps))
			)
			await saveModel(model, cli.flags.saveModelPath)
		})
		.catch(err => {
			console.error("Terrible error", err)
		})
}

// const model = buildSimpleRNNModel()
