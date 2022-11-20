import meow from "meow"
import tf from "@tensorflow/tfjs-node"
import {batchGenerator, exportSomeCsv, generator, numFeatures} from "./generator.js"
import {trainModel} from "./trainModel.js"
import {saveModel} from "./modelPersistence.js"

/**
 * Build a simpleRNN-based model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.LayersModel} A TensorFlow.js model consisting of a simpleRNN
 *   layer.
 */
export function buildSimpleRNNModel(inputShape: tf.Shape): tf.LayersModel {
	const model = tf.sequential()
	const rnnUnits = 32
	model.add(tf.layers.simpleRNN({units: rnnUnits, inputShape}))
	model.add(tf.layers.dense({units: 1}))
	return model
}

export const buildModel = ({numTimeSteps, numFeatures}: {numTimeSteps: number; numFeatures: number}) => {
	const inputShape = [numTimeSteps, numFeatures]

	const model = buildSimpleRNNModel(inputShape)

	model.compile({loss: "meanAbsoluteError", optimizer: "rmsprop"})
	model.summary()
	return model
}

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
				default: 4,
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
			},
			saveModelPath: {
				type: "string",
				default: "file:///tmp/rnn_test",
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
	for (const d of exportSomeCsv(cli.flags.outputRows)) {
		console.log(d)
	}
} else {
	const {lookBack, delay, loadModelPath} = cli.flags
	const model = !!loadModelPath ? tf.loadLayersModel(loadModelPath) : Promise.resolve(buildModel({numFeatures: numFeatures(), numTimeSteps: lookBack}))
	const coreGenerator = generator()

	let callback: any[] = []
	const {logDir, logUpdateFreq} = cli.flags
	if (!!logDir) {
		console.log(`Logging to tensorboard. ` + `Use the command below to bring up tensorboard server:\n` + `  tensorboard --logdir ${logDir}`)
		callback.push(
			tf.node.tensorBoard(logDir, {
				updateFreq: <"batch" | "epoch">logUpdateFreq,
			})
		)
	}

	const {epochs, batchSize} = cli.flags
	model
		.then(model => !loadModelPath ? trainModel(model, batchGenerator(lookBack, delay, batchSize, coreGenerator), epochs ?? 50, callback).then(async model => {
			const {saveModelPath} = cli.flags
			if (!!saveModelPath) {
				console.log("Saving model...")
				await saveModel(model, saveModelPath)
				return model
			}
		}) : model).then(model => {
			console.log(`Done with this model`)
		})
}

// const model = buildSimpleRNNModel()
