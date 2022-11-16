import meow from "meow"
import tf from "@tensorflow/tfjs-node"
import {exportSomeCsv} from "./generator.js"

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

const cli = meow(
	`
Usage: 
	$ yarn start [options]

	Options
		--outputDataset
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
		},
		importMeta: import.meta,
		allowUnknownFlags: false,
	}
)

if (cli.flags.outputDataset) {
	for (const d of exportSomeCsv(cli.flags.outputRows)) {
		console.log(d)
	}
}

// const model = buildSimpleRNNModel()
