import tf from "@tensorflow/tfjs-node"

// Test in repl:
// node --loader ts-node/esm.mjs
// # await import("./src/linearExample.ts")
// # z.getExampleTensor(5)

const a = 10
const b = 0.5

/** A line crossing origo at `a`, with a slope of `b` */
export const theFunction = (x: number) => a + x * b

/** Produce an array of {x, target} pairs */
export const getExampleValues = (count: number) =>
	Array.from({length: count}, (_, i) => Math.random())
		.map(r => r * 100)
		.map(x => ({x, target: theFunction(x)}))

/** Convert the  */
export const getExampleTensors = (count: number) =>
	tf.tensor2d(getExampleValues(count).map(({ x, target }) => [x, target]))
		.split(2, 1)		

/** Create a one-neuron sequential model - and compile it */
export const createModel = () => {
	const model = tf.sequential()
	model.add(
		tf.layers.dense({
			inputShape: [1],
			units: 1,
		})
	)
	model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });
	return model
}

/** Generate data and feed the model */
export const trainModel = (model: tf.Sequential, count = 20000, epochs = 5, batchSize = 5) => {
	const [values, targets] = getExampleTensors(count)
	return model.fit(values.reshape([count, 1]), targets.reshape([count, 1]), { epochs, batchSize })
}

/** Given a pre-trained model and a sample value, predict a value */
export const predict = (model: tf.Sequential, value: number) => {
	return model.predict(tf.tensor2d([value], [1, 1]))
}

export const printModel = (model: tf.Sequential) => { 
	console.log(`bias: ${(<any>model.layers[0]).bias.val.dataSync()}`)
	console.log(`kernel: ${(<any>model.layers[0]).kernel.val.dataSync()}`)
}

