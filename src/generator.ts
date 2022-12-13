import tf from "@tensorflow/tfjs-node"

export const counter = function* (max: number = -1) {
	let x = 0
	while (max < 0 || x < max) {
		yield x++
	}
}

const cycle = 60

type IData = {i: number; delta: number; target: number; event: 1 | -0}

const componentScale = 0.3
const baseLevel = 0.0  // to shift "expected" output up and down.  This affects the trained values.

export const generator = function* (max: number = -1) {
	let lastEvent = -3
	for (const i of counter(max)) {
		const impact = i === lastEvent + 2 ? componentScale : 0
		if (impact !== 0) {
		} else {
			if (i > lastEvent + 5 && 0.01 > Math.random()) {
				// If we want the probability to increase with time since last event: ((i - lastEvent) / 3000)
				lastEvent = i
			}
		}
		yield <IData>{
			i,
			delta: -1 + (2 * (i % cycle)) / cycle,
			event: lastEvent === i ? 1 : -1,
			target: baseLevel + componentScale * (Math.sin((i * Math.PI) / cycle) + Math.sin((3 * i * Math.PI) / cycle)) + impact,
		}
	}
}

export const sampleGenerator = function* (samplesToCollect: number, stream: Iterable<IData>) {
	let buffer: IData[] = []
	const bufferCutter = (newValue: IData, [oldest, ...rest]: IData[]) => (oldest && rest.length < samplesToCollect - 1 ? [oldest, ...rest, newValue] : [...rest, newValue])
	for (const data of stream) {
		buffer = bufferCutter(data, buffer)
		// Create a buffer, and assign a label to it.
		if (buffer.length === samplesToCollect) yield buffer
	}
}

const features: (keyof IData)[] = ["delta", "event"]

export const numFeatures = () => features.length

export const skipNonlinearsAndRandom = function*(iterator: Generator<IData[], void>, maxSkip: number) {
	let skip = -1
	while (true) {
		const {done, value: samples} = iterator.next()
		if (done || !samples)
			return
		if (skip-- > 0)
			continue
		const invalid = samples[samples.length -1].delta < samples[0].delta
		skip = Math.floor(Math.random() * maxSkip)
		if (!invalid)
			yield samples
	}
}

export const batchGenerator = function* (lookBack: number, delay: number, batchSize: number, stream: Iterable<IData>) {
	const iterator = skipNonlinearsAndRandom(sampleGenerator(lookBack + delay, stream)[Symbol.iterator](), cycle - lookBack)
	while (true) {
		const sampleTensor = tf.buffer([batchSize, lookBack, numFeatures()])
		const targetTensor = tf.buffer([batchSize, 1])

		for (const n of counter(batchSize)) {
			const {done, value: samples} = iterator.next()
			if (done) return

			// const actualSamples = delay > 0 ? samples.slice(0, lookBack) : samples
			const featuresBlock = samples.filter((_, i) => i < lookBack).map(s => features.map(f => s[f]))
			const {target} = samples[samples.length - 1]
			featuresBlock.forEach((sample, sampleIdx) => sample.forEach((value, colIdx) => sampleTensor.set(value, n, sampleIdx, colIdx)))
			targetTensor.set(target, n, 0)
		}
		// Verify that we are actually outputting the correct tensor size
		const results = {xs: sampleTensor.toTensor(), ys: targetTensor.toTensor()}
		yield results
	}
}

export const exportSomeCsv = function* (max: number) {
	yield `SeqNr,x,event,target`
	for (const {i, delta, target, event} of generator(max)) {
		yield `${i}, ${delta}, ${event}, ${target}`
	}
}
