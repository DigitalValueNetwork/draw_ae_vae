import jimp from 'jimp';

export type IImgProps = {imageHeight: number, imageWidth: number, channels: number}

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
		const inIndex = (i * imageWidth + j);
		const val = imageData[inIndex] * 255;
		buffer.set([Math.floor(val)], index++);
		buffer.set([Math.floor(val)], index++);
		buffer.set([Math.floor(val)], index++);
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
