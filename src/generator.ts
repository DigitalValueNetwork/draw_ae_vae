const counter = function* (max: number = -1) {
	let x = 0
	while (max < 0 || x < max) {
		yield x++
	}
}

const cycle = 30

type IData = {i: number; delta: number; target: number; event: 1 | -0}

export const generator = function* (max: number = -1) {
	let lastEvent = -3
	for (const i of counter(max)) {
		const impact = i === lastEvent + 2 ? 0.7 : 0
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
			target: Math.sin((i * Math.PI) / cycle) + Math.sin((3 * i * Math.PI) / cycle) + impact,
		}
	}
}

/*
export const batchGenerator = function* (lookBack: number, batchSize: number) {
	let buffer: IData[] = []
	const bufferCutter = (newValue: IData, [oldest, ...rest]: IData[]) => (rest.length < lookBack ? [oldest, ...rest, newValue] : [...rest, newValue])
	for (const data of generator()) {
		buffer = bufferCutter(data, buffer)
		// Create a buffer, and assign a label to it.
		if (buffer.length === lookBack) yield buffer
	}
} */

export const exportSomeCsv = function* (max: number) {
	yield `SeqNr,x,target`
	for (const {i, delta, target} of generator(max)) {
		yield `${i}, ${delta}, ${target}`
	}
}
