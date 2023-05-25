import * as tf from "@tensorflow/tfjs"

export const setupAnimation = (latentDim: number) => {
	const interpolate = (vectorA: number[], vectorB: number[], value: number) => [[vectorA, vectorB].map(v => tf.tensor(v, [1, latentDim]))].map(([a, b]) => a.add(b.sub(a).mul(value)))[0]

	const isArrayOfArrays = (arr: any[]): arr is number[][] => Array.isArray(arr[0])

	const s = (arr: number[][], index: number) => arr[Math.min(Math.floor(index), arr.length - 1)]

	const getAnimatedTensor = (input: (number | number[])[], animationIndex: number, override: number[], fullArray = override.length ? override : input) =>
		fullArray && fullArray.length
			? isArrayOfArrays(fullArray)
				? interpolate(s(fullArray, animationIndex), s(fullArray, animationIndex + 1), animationIndex - Math.floor(animationIndex))
				: tf.tensor(fullArray, [1, latentDim])
			: null

	return {getAnimatedTensor}
}
