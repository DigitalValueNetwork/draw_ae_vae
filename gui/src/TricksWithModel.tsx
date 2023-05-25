import React, {ChangeEvent, FormEvent, useRef, useState} from "react"

import * as tf from "@tensorflow/tfjs"
import {GenerateImage, latentDim} from "./GenerateImage"
import {range} from "radash"
import {LatentVectorSliders} from "./LatentVectorSliders"

/** Scale of the latent vectors, from -1 to 1 */
const scales = [1, 1, 1, 1, 1]

const comeUpWithDefault = () =>
	JSON.stringify([...range(1, 20)].map(() => [...range(1, latentDim)].map((_, i) => -scales[i] + (2 * Math.round(Math.sin((Math.random() * 3.14) / 2) * scales[i] * 1000)) / 1000))).replace(
		/(0{4,20}\d)|(9{4,20}\d)/g,
		""
	)

const interpolate = (vectorA: number[], vectorB: number[], value: number) => [[vectorA, vectorB].map(v => tf.tensor(v, [1, latentDim]))].map(([a, b]) => a.add(b.sub(a).mul(value)))[0]

const isArrayOfArrays = (arr: any[]): arr is number[][] => Array.isArray(arr[0])

const s = (arr: number[][], index: number) => arr[Math.min(Math.floor(index), arr.length - 1)]

const getAnimatedTensor = (fullArray: (number | number[])[], animationIndex: number) =>
	fullArray && fullArray.length
		? isArrayOfArrays(fullArray)
			? interpolate(s(fullArray, animationIndex), s(fullArray, animationIndex + 1), animationIndex - Math.floor(animationIndex))
			: tf.tensor(fullArray, [1, latentDim])
		: null

export const TricksWithModel = ({model}: {model: tf.LayersModel}) => {
	const [textArray, setTextArray] = useState<string>(comeUpWithDefault())
	const [validArray, setValidArray] = useState<boolean>(false)
	const [fullArray, setFullArray] = useState<(number | number[])[]>([])
	const [animationIndex, setAnimationIndex] = useState<number>(0)

	const textChange = (e: ChangeEvent) => {
		const text = (e.target as any).value
		setTextArray(text)
		try {
			const array = JSON.parse(text)
			if (!Array.isArray(array)) {
				throw Error("Not an array")
			}
			setValidArray(true)
		} catch {
			setValidArray(false)
		}
	}

	const handleSubmit = (e: FormEvent) => {
		const arr = JSON.parse(textArray)
		console.log(arr)
		// This timeout seems to be needed to have the visuals re-render
		setTimeout(() => setFullArray(arr), 10)

		e.preventDefault()
	}

	const regenerate = () => {
		setTextArray(comeUpWithDefault())
	}

	const requestRef = useRef()

	React.useEffect(() => {
		let startTime = +new Date()
		const animate =
			(start: number, stop: number, current = 0) =>
			() => {
				const passedTime = +new Date() - startTime
				const newCurrent = Math.min(stop, start + passedTime / 2000) // Math.min(stop, current + (stop - start) / 200)
				setAnimationIndex(newCurrent) // Another (better) way to do this is to pass a function here - it will receive a callback with the current state, which can then be updated.  https://css-tricks.com/using-requestanimationframe-with-react-hooks/

				requestRef.current = requestAnimationFrame(animate(start, stop, newCurrent)) as any
			}

		if (fullArray && fullArray.length && Array.isArray(fullArray[0])) {
			requestRef.current = requestAnimationFrame(animate(0, fullArray.length - 1)) as any
			return () => cancelAnimationFrame(requestRef.current as any)
		}
	}, [fullArray])

	const visualArray = getAnimatedTensor(fullArray, animationIndex)

	return (
		<div style={{display: "flex", flexDirection: "column"}}>
			<b>Model is loaded.</b>
			<form onSubmit={handleSubmit} style={{width: "250px"}}>
				<div style={{display: "flex", flexDirection: "column"}}>
					<input type="text" value={textArray} onChange={textChange} />
					{validArray && <input type="submit" value="Start latent animation" />}
				</div>
			</form>
			<button onClick={regenerate}>Generate Latent Vectors</button>
			<p>
				Animation Index: <span style={{width: "31px", display: "inline-block", textAlign: "end"}}>{(Math.round(animationIndex * 100) / 100).toPrecision(3)}</span>
			</p>
			{!!visualArray && <LatentVectorSliders latentVector={visualArray} />}
			{!!visualArray && <GenerateImage latentSpace={visualArray} model={model} />}
		</div>
	)
}
