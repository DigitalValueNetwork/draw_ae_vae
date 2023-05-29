import meow from "meow"
// import tfMain from "@tensorflow/tfjs-node"
// import tfGpu from "@tensorflow/tfjs-node-gpu"
// import {loadImages, imageProps as mnistImageProps, imagesFilePath as mnistImagesFilePath} from "./mnist-loading/data.js"
import {loadSeparateImages as loadImages, imageProps as srcImageProps, imagesFilePath as srcImagesFilePath} from "./individual-image-loading/data.js"
import {renderImageForTerminalPreview} from "./image/terminalImage.js"
import {ITrainModelSetup, train} from "./train.js"
import {imageChunks, imageChunkToFlat} from "./image/imageChunks.js"
import {saveModel} from "./persistence/saveModel.js"
import {loadTfjsGpu, loadTfjsNode} from "./tensorflowLoader.js"
import {setupDecoder, setupEncoder} from "./model.js"

const cli = meow(
	`
Usage: 
	$ yarn start [options]

	Options
		--useGpu use the GPU (if Cuda GPU is present)
		[--outputDataset Prints a CSV with the dataset]
		--lookBack Number of rows to include in input
		--delay Number of rows into the future to predict
		--logDir Tensorboard output
		--saveModelPath Path to save to, if overridden to empty string - no saving should happen
		--loadModelPath Path to model to load, including file - file:///tmp/rnn_test/model.json.
`,
	{
		flags: {
			/* had to use lowercase for some reason */
			useGpu: {
				type: "boolean",
				default: false,
			},
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
				default: "./output",
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

const saveModelSnapshot =
	(timestamp: number): ITrainModelSetup["saveModel"] =>
	async (type, epoch, model) => {
		await saveModel(model, cli.flags.saveModelPath, `${type}_${timestamp}_${epoch}`)
	}

if (cli.flags.outputDataset) {
	console.log("not implemented")
} else {
	//	const imageProps: IImgProps = { imageHeight: }

	const tensorflowPromise = cli.flags.useGpu ? loadTfjsGpu() : loadTfjsNode()

	tensorflowPromise.then(tensorflow =>
		loadImages(srcImagesFilePath, tensorflow.util as any, tensorflow as any)
			.then(async images => {
				console.log(await renderImageForTerminalPreview(images[Math.floor(images.length * Math.random())], srcImageProps))
				// console.log(await renderImageForTerminalPreview(images[100], mnistImageProps))
				// console.log(await renderImageForTerminalPreview(images[150], mnistImageProps))

				const imageDim = [150, 200, 3] as const
				const encoder = setupEncoder(tensorflow as any, imageDim)
				const decoder = setupDecoder(tensorflow as any)
				const startTime = +new Date()
				const modelSaver = saveModelSnapshot(startTime)

				const model = await train(
					// This crashes with RangeError: Maximum call stack... https://github.com/ReactiveX/rxjs/issues/651#issuecomment-153944205
					// When using a synchronous source with repeat, it will use recursion to trigger the new iterations, which will break.
					//   The solution is to use a subscribeOn with the asyncScheduler.
					imageChunkToFlat(imageChunks(images, cli.flags.batchSize)),
					cli.flags.epochs,
					async tensor => {
						console.log(await renderImageForTerminalPreview(tensor.dataSync() as Float32Array, srcImageProps))
					},
					{imageDim, encoder, decoder, saveModel: modelSaver},
					tensorflow as any
				)
				await saveModel(model, cli.flags.saveModelPath)
			})
			.catch(err => {
				console.error("Terrible error", err)
			})
	)
}

// const model = buildSimpleRNNModel()
