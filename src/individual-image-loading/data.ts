import path from "path"
import {readFile} from "fs/promises"
import {glob, globSync, globStream, globStreamSync, Glob} from "glob"
import {ITensorflow, Tensor3D} from "../tensorflowLoader"
import { IImgProps } from "../image/arrayToImage"

const DATASET_PATH = "/Users/jorgent/Downloads/fashion-extractor"
const TRAIN_IMAGES_GLOB = "*.jpg"

const IMAGE_HEIGHT = 200
const IMAGE_WIDTH = 150
const IMAGE_CHANNELS = 3
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

const loadIndividualImages = async (imgPath: string, _util: any, tf: ITensorflow) => {
	const iterator = new Glob(imgPath, {})
	const images: Float32Array[] = []
	for await (const file of iterator) {
		const imageBuffer = await readFile(file)
		tf.tidy(() => { 
			const tensor = <Tensor3D>tf.node.decodeImage(imageBuffer)
			if (JSON.stringify(tensor.shape) === JSON.stringify([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])) {
				// Return the image is Float32Array, even if it's already a nice tensor.  This is to keep the API similar to the mnist-example. Consider converting that to using tensors.
				images.push(tensor.div(tf.tensor1d([255])).dataSync() as Float32Array)
			}
			else { 
				console.log(`Bad dimensions, ${file} - ${JSON.stringify(tensor.shape)} - skipped it`)
			}	
		})
	}
	return images
}

const imagesFilePath = path.join(DATASET_PATH, TRAIN_IMAGES_GLOB)

const imageProps: IImgProps = {
	imageWidth: IMAGE_WIDTH,
	imageHeight: IMAGE_HEIGHT,
	channels: IMAGE_CHANNELS,
}

export {DATASET_PATH, IMAGE_FLAT_SIZE, imagesFilePath, imageProps, loadIndividualImages as loadSeparateImages}
