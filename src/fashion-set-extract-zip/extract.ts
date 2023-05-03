import {count, filter, map, take} from "rxjs"
import {generateHash, getFilenameFromLastArgument} from "../utils.js"
import {loadCSV} from "./loadCSV.js"
import {argv} from "process"
import {join as joinPath} from "path"

type IColumns = {
	id: string
	gender: "Men" | "Women" | "Boys" | "Girls" | "Unisex"
	masterCategory: "Apparel" | "Footwear" | "Accessories"
	subCategory: "Topwear" | "Saree" | "Dress" | "Bottomwear" | "Topwear" | "Dress" | "Innerwear" | "Flip Flops" | "Shoes" | "Sandal"
	articleType: string
	baseColour: string
	season: string
	year: string
	usage: string
	productDisplayName: string
}

// Keeps the aspect radio of the original images
const targetDimensions = "150x200" 

const sourceZipFile = getFilenameFromLastArgument(argv)
const targetFolder = getFilenameFromLastArgument(argv, 1, true) || "~/Downloads/fashion-extractor"

const generateScript = ({id, gender, subCategory}: IColumns) => {
	const targetFileName = `${id}_${gender}_${subCategory}.jpg`
	const targetFullPath = joinPath(targetFolder, targetFileName)
	const extractedSourceFile = joinPath(targetFolder, `${id}.jpg`)
	return `if [ ! -f ${targetFullPath} ]
then
	unzip -j ${sourceZipFile} fashion-dataset/images/${id}.jpg -d ${targetFolder}
	convert ${extractedSourceFile} -resize ${targetDimensions} ${targetFullPath}
	rm ${extractedSourceFile}
else
	let SKIP_COUNTER++
fi
`
}

const genders: IColumns["gender"][] = ["Men", "Women"]
const subCategories: IColumns["subCategory"][] = ["Topwear", "Bottomwear"]

/** Selects some entries based on the hash of the id */
const deterministicFilter =
	(cutChar: string) =>
	({id}: IColumns): boolean =>
		generateHash(id, 1) < cutChar

const loadStyles = (file: string) =>
	loadCSV<IColumns>(file).pipe(
		filter(({gender, subCategory}) => genders.includes(gender) && subCategories.includes(subCategory)),
		filter(deterministicFilter("7")),
		map(generateScript),
		take(1000)
	)

console.log(`#!/bin/bash
SKIP_COUNTER=0
`)
loadStyles("/Users/jorgent/Downloads/styles.csv")
	.forEach(console.log)
	.then(() => console.log(`printf "SkipCount = %d\\n" $SKIP_COUNTER`))
