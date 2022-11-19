import tf from "@tensorflow/tfjs-node"

/**
 * Train a model on the Jena weather data.
 *
 * @param {tf.LayersModel} model A compiled tf.LayersModel object. It is
 *   expected to have a 3D input shape `[numExamples, timeSteps, numFeatures].`
 *   and an output shape `[numExamples, 1]` for predicting the temperature
 * value.
 * @param {JenaWeatherData} jenaWeatherData A JenaWeatherData object.
 * @param {boolean} normalize Whether to used normalized data for training.
 * @param {boolean} includeDateTime Whether to include date and time features
 *   in training.
 * @param {number} lookBack Number of look-back time steps.
 * @param {number} step Step size used to generate the input features.
 * @param {number} delay How many steps in the future to make the prediction
 *   for.
 * @param {number} batchSize batchSize for training.
 * @param {number} epochs Number of training epochs.
 * @param {tf.Callback | tf.CustomCallbackArgs} customCallback Optional callback
 *   to invoke at the end of every epoch. Can optionally have `onBatchEnd` and
 *   `onEpochEnd` fields.
 */
export const trainModel = async (
	model: tf.LayersModel,
	// jenaWeatherData: any,
	generator: Generator<{xs: any, ys: any}>,
	epochs: number,
	customCallback: tf.CustomCallbackArgs | tf.CustomCallbackArgs[]
) => {
	// const trainShuffle = true
	const trainDataset = tf.data
		.generator(() => generator)
		// .prefetch(8)
	//const evalShuffle = false
	const valDataset = tf.data.generator(function*() {
		let trainingBatchesAllowed = 600
		for (const f of generator) {
			yield f
			if (trainingBatchesAllowed-- <= 0)
				return
		}
	})

	await model.fitDataset(trainDataset, {
		batchesPerEpoch: 500,
		validationBatches: 100,
		epochs,
		callbacks: customCallback,
		validationData: valDataset,
	})
	return model
}
