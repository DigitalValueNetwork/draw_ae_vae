import React, {ChangeEvent, FormEvent, useState} from "react"

import * as tf from "@tensorflow/tfjs"
import { GenerateImage } from "./GenerateImage"

export const TricksWithModel = ({model}: {model: tf.LayersModel}) => {
	const [textArray, setTextArray] = useState<string>("")
	const [validArray, setValidArray] = useState<boolean>(false)
	const [visualArray, setVisualArray] = useState<number[]>([])

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
		setVisualArray(arr)
		
		e.preventDefault()
	}

	return (
		<p>
			Model is now loaded!
			<form onSubmit={handleSubmit}>
				<div style={{display: "flex", flexDirection: "column"}}>
					<input type="text" value={textArray} onChange={textChange} />
					{validArray && <input type="submit" value="Try it!" />}
				</div>
			</form>
			{
				visualArray && visualArray.length && 
					<GenerateImage latentSpace={visualArray} model={model}/>
			}
		</p>
	)
}
