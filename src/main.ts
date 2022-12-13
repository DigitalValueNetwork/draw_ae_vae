import meow from "meow"
import tf from "@tensorflow/tfjs-node"
import {batchGenerator, exportSomeCsv, generator, numFeatures, counter} from "./generator.js"
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
	const rnnUnits = 5
	model.add(tf.layers.simpleRNN({units: rnnUnits, inputShape, activation: "tanh"}))
//	model.add(tf.layers.dense({units: 50, activation: "tanh"}))
	model.add(tf.layers.dense({units: 1, activation: "tanh"}))
	return model
}

export const compileModel = (model: tf.LayersModel) => {
	// Maybe the optimizer is not the best?  Tried adam. I guess it can't be the main reason for the troubles.
	// What effect does ...Squared... have compared to ...Absolute...?  Squared actually gives lower error values [-1, 1], but bigger diffs still gives a higher error.
	model.compile({loss: "meanSquaredError", optimizer: "rmsprop"})
	return model
}

export const buildModel = ({numTimeSteps, numFeatures}: {numTimeSteps: number; numFeatures: number}) => {
	const inputShape = [numTimeSteps, numFeatures]

	const model = compileModel(buildSimpleRNNModel(inputShape))

	model.summary()
	return model
}

export const loadModel = async (path: string) => compileModel(await tf.loadLayersModel(path))

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
	const model = !!loadModelPath ? loadModel(loadModelPath) : Promise.resolve(buildModel({numFeatures: numFeatures(), numTimeSteps: lookBack}))
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

	// Maybe we must normalize?
	//    The output shows signs of learning when the curve crosses the center, in the big way.
	//    Result: No sign of an improvement
	// Maybe the optimizer is not cool?
	//    Result: Same result with different optimizers
	// Maybe the loss function is not cool, since the average loss might look OK here?
	//    Result: Different loss functions yields different results - and some errors.  Did use meanAbsoluteError for a while, instead of meanSquareError.   This did not affect the results
	//    Perhaps we are using the loss function in a weird way.  Should use tf.losses.meanSquared...
	// Output the inputs and outputs, and see what they look like
	//    Result: tried to look at the loss function outputs, and they look as expected
	// Shifting the dataset did not help, but it moved the estimations to the new mean.
	// Adjusting the number of layers, and nodes did not help
	// It would be interesting to test the logic without the rnn, but unfortunately, that would require rewriting all the input and testing logic
	//     As it requires only a single input
	// Maybe the RNN has trouble with the non-continuos input? That it breaks on the shifts from 1 to 0/-1
	//     Could try to reverse the delta when it reaches 1, instead of restarting it.
	// Try:  Skip the sinus, try with just the signal (event)  - any response?  Maybe a linear link to the 'delta'.
	// Would be nice to try with some non-linear layer before the RNN, but then we would have to deal with the dataformat again
	// It should be possible to debug this stuff, see what information is going into the layers (But most likely, the RNN does not support the data?)
	// Try: Reverse the "angle" argument, "delta" to avoid shifts in the data that has no effect on the output.
	//     Reversing is most likely a problem for the RNN as the order of things is somewhat important.
	//     Instead - try to avoid having a delta-reset in the datasets.  Skip past it.
	// SOLUTION:  Bug in data generation - the output was not related to the input, just seemed to be.  This was evident, but overlooked, in the spreadsheet.
	// Next up: OK: Export the inputs and the outputs - see if it learns the outliers.
	// Next up: Watch train progress in the tensorflow inspector thing.  Tensorboard is not available without python and tensorflow
	// Next up: Experiment with meta parameters.

	const {epochs, batchSize} = cli.flags
	model
		.then(model =>
			!loadModelPath
				? trainModel(model, batchGenerator(lookBack, delay, batchSize, coreGenerator), epochs ?? 50, callback).then(async model => {
						const {saveModelPath} = cli.flags
						if (!!saveModelPath) {
							console.log("Saving model...")
							await saveModel(model, saveModelPath)
							return model
						}
				  })
				: model
		)
		.then(async model => {
			// const dataSet = tf.data.generator(() => batchGenerator(lookBack, delay, batchSize, coreGenerator))
			let iterations = 5

			console.log(`delta, event, expected, predicted`)
			for (const item of batchGenerator(lookBack, delay, batchSize, coreGenerator, false)) {
				const result = model?.predict(item.xs) as tf.Tensor
				const inputs = <number[][][]>item.xs.arraySync()
				const predicted = [...result.dataSync()]
				const expected = [...item.ys.dataSync()]
				// Print the generated data vs the expected data
				for (const i of counter(predicted.length)) {
					const inputBatch = inputs[i]
					const lastRow = inputBatch[inputBatch.length - 1]
					const [delta, event] = lastRow
					
					console.log(`${delta}, ${event}, ${expected[i]}, ${predicted[i]}`)
				}
				if (iterations-- <= 0)
					break
			}
		})
}

// const model = buildSimpleRNNModel()
