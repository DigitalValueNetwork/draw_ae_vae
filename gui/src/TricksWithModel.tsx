import React, {ChangeEvent, FormEvent, useRef, useState} from "react"

import * as tf from "@tensorflow/tfjs"
import {GenerateImage, latentDim} from "./GenerateImage"
import {range} from "radash"

const comeUpWithDefault = () => 
	JSON.stringify([...range(1, 5)].map(() => [...range(1, latentDim)].map(() => Math.round(Math.random() * 1000) / 1000)))

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
		// Take the array to animate on here
		// Store new progresses based array indices, but with decimal values - 

		// create an isAnimateSet function 
		// create an interpolateTensor function:  a.add(b.sub(a).mul(0.99))
		const newCurrent = Math.min(stop, current + (stop - start) / 100)
		setAnimationIndex(newCurrent) // Another (better) way to do this is to pass a function here - it will receive a callback with the current state, which can then be updated.  https://css-tricks.com/using-requestanimationframe-with-react-hooks/

		requestRef.current = requestAnimationFrame(animate(start, stop, newCurrent)) as any
	}

		if (fullArray && fullArray.length && Array.isArray(fullArray[0])) {
			requestRef.current = requestAnimationFrame(animate(0, fullArray.length)) as any
			return () => cancelAnimationFrame(requestRef.current as any)	
		}
	}, [fullArray])

	const visualArray = fullArray && fullArray.length ? ((Array.isArray(fullArray[0]) ? fullArray[Math.floor(animationIndex)] : fullArray) as number[]) : null

	return (
		<div style={{display: "flex", flexDirection: "column"}}>
			Model is now loaded!
			<form onSubmit={handleSubmit} style={{width: "250px"}}>
				<div style={{display: "flex", flexDirection: "column"}}>
					<input type="text" value={textArray} onChange={textChange} />
					{validArray && <input type="submit" value="Try it!" />}
				</div>
			</form>
			<p>Animation Index: {Math.round(animationIndex * 100) / 100}</p>
			{!!(visualArray && visualArray.length) && <GenerateImage latentSpace={visualArray} model={model} />}
		</div>
	)
}
