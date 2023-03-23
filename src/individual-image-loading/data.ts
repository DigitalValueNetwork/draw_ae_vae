import path from "path"
import {readFile} from "fs/promises"
import {glob, globSync, globStream, globStreamSync, Glob} from "glob"
import {ITensorflow, Tensor3D} from "../tensorflowLoader"
import { IImgProps } from "../image/arrayToImage"

const DATASET_PATH = "./dataset"
const TRAIN_IMAGES_GLOB = "mangled_*.jpg"

const IMAGE_HEIGHT = 128
const IMAGE_WIDTH = 128
const IMAGE_CHANNELS = 3
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

const loadIndividualImages = async (imgPath: string, _util: any, tf: ITensorflow) => {
	const iterator = new Glob(imgPath, {})
	const images: Float32Array[] = []
	for await (const file of iterator) {
		const imageBuffer = await readFile(file)
		const tensor = <Tensor3D>tf.node.decodeImage(imageBuffer)
		// Return the image is Float32Array, even if it's already a nice tensor.  This is to keep the API similar to the mnist-example. Consider converting that to using tensors.
		images.push(tensor.dataSync() as Float32Array)
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
