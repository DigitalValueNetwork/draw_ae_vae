import jimp from 'jimp';

export type IImgProps = {imageHeight: number, imageWidth: number, channels: number}

/** Produce RGB values from row and columns. Depending on the number of source image channels, this might be separate values or just the same grayscale image. */
const readSrcValues = (row: number, col: number, imageData: Float32Array, { imageWidth, channels }: IImgProps) =>
	channels === 1 ? new Array(3).map(() => imageData[row * imageWidth + col]) :
		new Array(3).map((_, i) => imageData[channels * (row * imageWidth + (col + i))])

/**
 * Convert an image represented as a typed array, with 1 channel, to a JIMP RGB image.
 *
 * @param {Float32Array} imageData
 *
 * @returns Jimp object representing image.
 */
async function arrayToJimp(imageData: Float32Array, {imageHeight, imageWidth, channels}: IImgProps): Promise<jimp> {
	const bufferLen = imageHeight * imageWidth * 4;
	const buffer = new Uint8Array(bufferLen);
  
	let index = 0;
	for (let i = 0; i < imageHeight; ++i) {
	  for (let j = 0; j < imageWidth; ++j) {
		// const inIndex = (i * imageWidth + j);
		const pixelValues = readSrcValues(i, j, imageData, {imageWidth, channels} as any) // imageData[inIndex] * 255;
		for (const pixelValue of pixelValues)
			buffer.set([Math.floor(pixelValue)], index++);
		// Alpha:
		buffer.set([255], index++);
	  }
	}
  
	return new Promise((resolve, reject) => {
	  new jimp(
		  {data: buffer, width: imageWidth, height: imageHeight},
		  (err: any, img: jimp) => {
			if (err) {
			  reject(err);
			} else {
			  resolve(img);
			}
		  });
	});
  }

export const pixelArrayToPngByteBuffer = (imageData: Float32Array, imgProps: IImgProps) => 
	arrayToJimp(imageData, imgProps)
		.then(imageAsJimp => imageAsJimp.getBufferAsync(jimp.MIME_PNG))
