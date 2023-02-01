import React, {ChangeEvent, FormEvent, useRef, useState} from "react"

import * as tf from "@tensorflow/tfjs"
import {GenerateImage, latentDim} from "./GenerateImage"
import {range} from "radash"

const scales = [
	0, 0, 1.3707246780395508, 1.7865811586380005, 2.6706178188323975, 1.8973675966262817, 0, 2.3221395015716553, 0, 2.366811990737915, 2.2038114070892334, 2.3543779850006104, 2.6084847450256348,
	1.95780611038208, 0, 2.3057286739349365,
]

const comeUpWithDefault = () => JSON.stringify([...range(1, 5)].map(() => [...range(1, latentDim)].map((_, i) => Math.round(Math.sin(Math.random() * 3.14/2) * scales[i] * 1000) / 1000)))

const interpolate = (vectorA: number[], vectorB: number[], value: number) => [[vectorA, vectorB].map(v => tf.tensor(v, [1, latentDim]))].map(([a, b]) => a.add(b.sub(a).mul(value)))[0]

const isArrayOfArrays = (arr: any[]): arr is number[][] => Array.isArray(arr[0])

const s = (arr: number[][], index: number) => arr[Math.min(Math.floor(index), arr.length - 1)]

const getAnimatedTensor = (fullArray: (number | number[])[], animationIndex: number) =>
	fullArray && fullArray.length
		? isArrayOfArrays(fullArray)
			? interpolate(s(fullArray, animationIndex) , s(fullArray, animationIndex + 1), animationIndex - Math.floor(animationIndex))
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

	const requestRef = useRef()

	React.useEffect(() => {
		const animate =
			(start: number, stop: number, current = 0) =>
			() => {
				const newCurrent = Math.min(stop, current + (stop - start) / 100)
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
			Model is now loaded!
			<form onSubmit={handleSubmit} style={{width: "250px"}}>
				<div style={{display: "flex", flexDirection: "column"}}>
					<input type="text" value={textArray} onChange={textChange} />
					{validArray && <input type="submit" value="Try it!" />}
				</div>
			</form>
			<p>
				Animation Index: <span style={{width: "31px", display: "inline-block", textAlign: "end"}}>{(Math.round(animationIndex * 100) / 100).toPrecision(3)}</span>
			</p>
			{!!visualArray && <GenerateImage latentSpace={visualArray} model={model} />}
		</div>
	)
}
